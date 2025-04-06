import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class ImageProcessor:
    @staticmethod
    def resize_image(image, target_size=(224, 224)):
        """이미지 크기 조정"""
        return cv2.resize(image, target_size)
    
    @staticmethod
    def preprocess_for_training(image_path, target_size=(224, 224)):
        """학습을 위한 이미지 전처리"""
        # 이미지 로드 및 전처리
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # BGR에서 RGB로 변환
        image = ImageProcessor.resize_image(image, target_size)
        image = img_to_array(image)
        image = preprocess_input(image)  # MobileNetV2 전처리
        
        return image
    
    @staticmethod
    def preprocess_for_prediction(image_path, target_size=(224, 224)):
        """예측을 위한 이미지 전처리"""
        # 이미지 로드 및 전처리
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = ImageProcessor.resize_image(image, target_size)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)  # 배치 차원 추가
        image = preprocess_input(image)  # MobileNetV2 전처리
        
        return image
    
    @staticmethod
    def load_and_preprocess_dataset(data_dir, target_size=(224, 224)):
        """데이터셋 로드 및 전처리"""
        images = []
        labels = []
        label_names = []
        
        # 각 클래스 폴더 순회
        for i, class_name in enumerate(sorted(os.listdir(data_dir))):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            label_names.append(class_name)
            print(f"클래스 로드 중: {class_name}, 인덱스: {i}")
            
            # 클래스 내의 모든 이미지 순회
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if not img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                try:
                    img = ImageProcessor.preprocess_for_training(img_path, target_size)
                    images.append(img)
                    labels.append(i)
                except Exception as e:
                    print(f"이미지 로드 실패: {img_path}, 오류: {e}")
        
        return np.array(images), np.array(labels), label_names