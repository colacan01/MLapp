from django.urls import path
from . import views
from django.conf import settings
from django.conf.urls.static import static
from .routing import websocket_urlpatterns

app_name = 'ml_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('upload/', views.upload_view, name='upload'),
    path('upload/success/', views.upload_success, name='upload_success'),
    path('ajax-upload/', views.ajax_upload, name='ajax_upload'),
    path('training/', views.training_view, name='training'),
    path('training/status/<str:training_id>/', views.training_status, name='training_status'),
    path('test/', views.test_view, name='test'),
    path('predict/', views.predict_image, name='predict_image'),
    path('training-list/', views.training_list_view, name='training_list'),
    path('models/', views.model_list_view, name='model_list'),
    path('models/toggle/<int:model_id>/', views.toggle_model_status, name='toggle_model'),
]

urlpatterns += websocket_urlpatterns

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)