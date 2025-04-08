from django.contrib import admin
from django.utils.html import format_html
from .models import Label, ImageCategory, TrainingImage, TrainedModel

@admin.register(Label)
class LabelAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_at')
    search_fields = ('name',)
    ordering = ('name',)

@admin.register(ImageCategory)
class ImageCategoryAdmin(admin.ModelAdmin):
    list_display = ('name', 'image_count', 'created_at')
    search_fields = ('name',)
    ordering = ('name',)
    
    def image_count(self, obj):
        return obj.count_images()
    image_count.short_description = '이미지 수'

@admin.register(TrainingImage)
class TrainingImageAdmin(admin.ModelAdmin):
    list_display = ('id', 'category', 'image_preview', 'uploaded_at')
    list_filter = ('category', 'uploaded_at')
    search_fields = ('category__name',)
    ordering = ('-uploaded_at',)
    
    def image_preview(self, obj):
        if obj.image:
            return format_html('<img src="{}" width="50" height="50" style="object-fit: cover;" />', obj.image.url)
        return "이미지 없음"
    image_preview.short_description = '이미지 미리보기'

@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'accuracy_percentage', 'is_active', 'created_at')
    list_filter = ('is_active', 'created_at')
    search_fields = ('name',)
    ordering = ('-created_at',)
    list_editable = ('is_active',)
    
    def accuracy_percentage(self, obj):
        return f"{obj.accuracy * 100:.2f}%"
    accuracy_percentage.short_description = '정확도'
