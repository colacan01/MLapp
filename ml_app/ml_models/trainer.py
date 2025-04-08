import os
import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
import threading
from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer

class TrainingProgressCallback(Callback):
    def __init__(self, channel_name):
        super().__init__()
        self.channel_name = channel_name
        self.channel_layer = get_channel_layer()
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        async_to_sync(self.channel_layer.group_send)(
            self.channel_name,
            {
                'type': 'training_update',
                'message': {
                    'epoch': epoch + 1,
                    'loss': float(logs.get('loss', 0)),
                    'accuracy': float(logs.get('accuracy', 0)),
                    'val_loss': float(logs.get('val_loss', 0)),
                    'val_accuracy': float(logs.get('val_accuracy', 0))
                }
            }
        )
    
    def send_error(self, error_message):
        """오류 메시지를 클라이언트에 전송"""
        async_to_sync(self.channel_layer.group_send)(
            self.channel_name,
            {
                'type': 'training_error',
                'message': {
                    'error': error_message
                }
            }
        )

class ModelTrainer:
    def __init__(self, data_dir, model_dir, img_width=224, img_height=224):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.img_width = img_width
        self.img_height = img_height
        self.model = None
        self.training_thread = None
        
    def build_model(self, num_classes):
        # MobileNetV2를 기반 모델로 사용
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_width, self.img_height, 3)
        )
        
        # 기반 모델 레이어 동결
        base_model.trainable = False
        
        # 새로운 분류기 추가
        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(num_classes, activation='softmax')
        ])
        
        # 모델 컴파일
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, batch_size=32, validation_split=0.2):
        # 디렉토리 내용 확인을 위한 로깅 추가
        print(f"데이터 디렉토리: {self.data_dir}")
        try:
            print(f"디렉토리 내용: {os.listdir(self.data_dir)}")
            
            # 각 하위 디렉토리 내 이미지 수 확인
            for class_dir in os.listdir(self.data_dir):
                class_path = os.path.join(self.data_dir, class_dir)
                if os.path.isdir(class_path):
                    img_count = len([f for f in os.listdir(class_path) 
                                  if os.path.isfile(os.path.join(class_path, f)) and 
                                  f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    print(f"클래스 '{class_dir}': 이미지 {img_count}개")
        except Exception as e:
            print(f"디렉토리 확인 중 오류: {e}")
        
        # 데이터 증강 설정
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        # 학습 데이터셋
        try:
            train_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=batch_size,
                class_mode='categorical',
                subset='training'
            )
            print(f"학습 데이터셋: {train_generator.samples}개 이미지, {len(train_generator.class_indices)}개 클래스")
        except Exception as e:
            print(f"학습 데이터셋 생성 중 오류: {e}")
            raise ValueError(f"학습 데이터셋을 생성할 수 없습니다: {e}")
        
        # 검증 데이터셋
        try:
            validation_generator = train_datagen.flow_from_directory(
                self.data_dir,
                target_size=(self.img_width, self.img_height),
                batch_size=batch_size,
                class_mode='categorical',
                subset='validation'
            )
            print(f"검증 데이터셋: {validation_generator.samples}개 이미지")
        except Exception as e:
            print(f"검증 데이터셋 생성 중 오류: {e}")
            raise ValueError(f"검증 데이터셋을 생성할 수 없습니다: {e}")
        
        # 데이터셋 유효성 검사
        if train_generator.samples == 0:
            raise ValueError("학습 데이터셋이 비어 있습니다. 이미지를 업로드했는지 확인하세요.")
        
        if validation_generator.samples == 0:
            raise ValueError("검증 데이터셋이 비어 있습니다. validation_split 값을 줄이거나 더 많은 이미지를 업로드하세요.")
        
        # 클래스 인덱스를 JSON 파일로 저장
        class_indices = train_generator.class_indices
        class_names = {v: k for k, v in class_indices.items()}
        
        with open(os.path.join(self.model_dir, 'class_indices.json'), 'w') as f:
            json.dump(class_names, f)
        
        return train_generator, validation_generator, len(class_indices)
    
    def train_model(self, epochs=10, batch_size=32, learning_rate=0.001, validation_split=0.2, channel_name=None):
        # 콜백 객체를 미리 생성 (오류 전송에 사용하기 위함)
        progress_callback = None
        if channel_name:
            progress_callback = TrainingProgressCallback(channel_name)
        
        try:
            # 데이터 준비
            train_generator, validation_generator, num_classes = self.prepare_data(batch_size, validation_split)
            
            # 모델이 생성되지 않았다면 생성
            if self.model is None:
                self.build_model(num_classes)
            
            # 옵티마이저 학습률 설정
            self.model.optimizer.learning_rate = learning_rate
            
            # 콜백 정의
            callbacks = [
                ModelCheckpoint(
                    filepath=os.path.join(self.model_dir, 'model_checkpoint.h5'),
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                )
            ]
            
            # WebSocket을 통한 진행 상황 업데이트를 위한 콜백 추가
            if progress_callback:
                callbacks.append(progress_callback)
            
            # steps_per_epoch와 validation_steps 계산 시 안전하게 처리
            steps_per_epoch = max(1, train_generator.samples // batch_size)
            validation_steps = max(1, validation_generator.samples // batch_size)
            
            print(f"학습 시작: 이미지 {train_generator.samples}개, 클래스 {num_classes}개")
            print(f"steps_per_epoch: {steps_per_epoch}, validation_steps: {validation_steps}")
            
            # 모델 훈련
            history = self.model.fit(
                train_generator,
                steps_per_epoch=steps_per_epoch,
                epochs=epochs,
                validation_data=validation_generator,
                validation_steps=validation_steps,
                callbacks=callbacks
            )
            
            # 모델 저장
            self.model.save(os.path.join(self.model_dir, 'final_model.h5'))
            
            # 훈련 히스토리 저장
            with open(os.path.join(self.model_dir, 'training_history.json'), 'w') as f:
                history_dict = {key: [float(x) for x in values] for key, values in history.history.items()}
                json.dump(history_dict, f)
            
            return history.history, self.model
        
        except Exception as e:
            error_message = str(e)
            print(f"학습 중 오류 발생: {error_message}")
            
            # 오류 정보를 파일로 저장
            with open(os.path.join(self.model_dir, 'error_log.txt'), 'w') as f:
                f.write(f"학습 오류: {error_message}\n")
            
            # WebSocket을 통해 오류 전송
            if progress_callback:
                progress_callback.send_error(error_message)
            
            # 오류를 다시 발생시켜 호출자에게 전파
            raise
    
    def start_training_thread(self, epochs=10, batch_size=32, learning_rate=0.001, validation_split=0.2, channel_name=None):
        """백그라운드 스레드에서 훈련 시작"""
        self.training_thread = threading.Thread(
            target=self.train_model,
            kwargs={
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'validation_split': validation_split,
                'channel_name': channel_name
            }
        )
        self.training_thread.daemon = True
        self.training_thread.start()
    
    @staticmethod
    def load_trained_model(model_path, class_indices_path):
        """저장된 모델 로드"""
        model = load_model(model_path)
        
        with open(class_indices_path, 'r') as f:
            class_indices = json.load(f)
        
        return model, class_indices
    
    @staticmethod
    def predict(model, image, class_indices):
        """이미지 예측"""
        predictions = model.predict(image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        
        class_name = class_indices.get(str(predicted_class_idx), "Unknown")
        
        return class_name, confidence