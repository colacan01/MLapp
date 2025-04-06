import os
import uuid
import json
from django.shortcuts import render, redirect
from django.conf import settings
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .forms import ImageUploadForm, TrainingForm
from .models import ImageCategory, TrainingImage, TrainedModel
from .ml_models.image_processor import ImageProcessor
from .ml_models.trainer import ModelTrainer

def index(request):
    return render(request, 'ml_app/index.html')

def upload_view(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # 라벨 가져오기 또는 생성하기
            label = form.cleaned_data['label']
            category, created = ImageCategory.objects.get_or_create(name=label)
            
            # 이미지 저장
            image = form.cleaned_data['image']
            training_image = TrainingImage(category=category, image=image)
            training_image.save()
            
            return redirect('upload_success')
    else:
        form = ImageUploadForm()
    
    categories = ImageCategory.objects.all()
    return render(request, 'ml_app/upload.html', {'form': form, 'categories': categories})

def upload_success(request):
    return render(request, 'ml_app/upload_success.html')

@csrf_exempt
def ajax_upload(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image = request.FILES['image']
        label = request.POST.get('label', '')
        
        if not label:
            return JsonResponse({'status': 'error', 'message': '라벨이 필요합니다'})
        
        # 라벨 가져오기 또는 생성하기
        category, created = ImageCategory.objects.get_or_create(name=label)
        
        # 이미지 저장
        training_image = TrainingImage(category=category, image=image)
        training_image.save()
        
        return JsonResponse({
            'status': 'success',
            'message': '이미지가 성공적으로 업로드되었습니다',
            'image_url': training_image.image.url
        })
    
    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다'})

def training_view(request):
    if request.method == 'POST':
        form = TrainingForm(request.POST)
        if form.is_valid():
            # 훈련 파라미터 가져오기
            epochs = form.cleaned_data['epochs']
            batch_size = form.cleaned_data['batch_size']
            learning_rate = form.cleaned_data['learning_rate']
            validation_split = form.cleaned_data['validation_split']
            
            # 훈련 ID 생성
            training_id = str(uuid.uuid4())
            
            # 모델 디렉토리 생성
            model_dir = os.path.join(settings.MEDIA_ROOT, 'models', training_id)
            os.makedirs(model_dir, exist_ok=True)
            
            # 학습 데이터 디렉토리
            data_dir = os.path.join(settings.MEDIA_ROOT, 'training_data')
            
            # 모델 훈련기 생성
            trainer = ModelTrainer(data_dir, model_dir)
            
            # 백그라운드에서 훈련 시작
            trainer.start_training_thread(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                validation_split=validation_split,
                channel_name=f'training_{training_id}'
            )
            
            return render(request, 'ml_app/training_progress.html', {
                'training_id': training_id
            })
    else:
        form = TrainingForm()
    
    # 현재 사용 가능한 카테고리(라벨) 목록
    categories = ImageCategory.objects.all()
    
    return render(request, 'ml_app/training.html', {
        'form': form,
        'categories': categories
    })

def training_status(request, training_id):
    # 학습 상태 정보를 위한 JSON 파일 경로
    history_path = os.path.join(settings.MEDIA_ROOT, 'models', training_id, 'training_history.json')
    
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            history = json.load(f)
            
        # 모델 저장
        model_path = os.path.join(settings.MEDIA_ROOT, 'models', training_id, 'final_model.h5')
        if os.path.exists(model_path):
            # 모델 정보 데이터베이스에 저장
            accuracy = max(history.get('val_accuracy', [0]))
            model = TrainedModel.objects.create(
                name=f"Model-{training_id[:8]}",
                file_path=model_path,
                accuracy=accuracy,
                is_active=True
            )
            
            # 기존 활성 모델 비활성화
            TrainedModel.objects.exclude(pk=model.pk).update(is_active=False)
            
            return JsonResponse({
                'status': 'completed',
                'history': history,
                'model_id': model.id
            })
        
        return JsonResponse({
            'status': 'in_progress',
            'history': history
        })
    
    return JsonResponse({
        'status': 'waiting',
        'message': '학습이 아직 시작되지 않았거나 진행 중입니다'
    })

def test_view(request):
    # 활성화된 모델 가져오기
    model = TrainedModel.objects.filter(is_active=True).first()
    
    return render(request, 'ml_app/test.html', {'model': model})

@csrf_exempt
def predict_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        # 활성화된 모델 가져오기
        model_obj = TrainedModel.objects.filter(is_active=True).first()
        
        if not model_obj:
            return JsonResponse({
                'status': 'error',
                'message': '활성화된 모델이 없습니다. 먼저 모델을 학습시켜주세요.'
            })
        
        # 이미지 저장
        image = request.FILES['image']
        path = default_storage.save(f'uploads/{image.name}', ContentFile(image.read()))
        image_path = os.path.join(settings.MEDIA_ROOT, path)
        
        # 클래스 인덱스 파일 경로
        model_dir = os.path.dirname(model_obj.file_path)
        class_indices_path = os.path.join(model_dir, 'class_indices.json')
        
        try:
            # 이미지 전처리
            processed_image = ImageProcessor.preprocess_for_prediction(image_path)
            
            # 모델 로드 및 예측
            model, class_indices = ModelTrainer.load_trained_model(
                model_obj.file_path, 
                class_indices_path
            )
            
            class_name, confidence = ModelTrainer.predict(model, processed_image, class_indices)
            
            return JsonResponse({
                'status': 'success',
                'class': class_name,
                'confidence': confidence,
                'image_url': default_storage.url(path)
            })
            
        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f'예측 중 오류 발생: {str(e)}'
            })
        finally:
            # 업로드된 임시 이미지 정리 (선택 사항)
            # default_storage.delete(path)
            pass
    
    return JsonResponse({'status': 'error', 'message': '잘못된 요청입니다'})