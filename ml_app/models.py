from django.db import models
import os

class Label(models.Model):
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name

class ImageCategory(models.Model):
    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return self.name
    
    def count_images(self):
        return TrainingImage.objects.filter(category=self).count()

class TrainingImage(models.Model):
    image = models.ImageField(upload_to='training_data/')
    category = models.ForeignKey(ImageCategory, on_delete=models.CASCADE, related_name='images')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"{self.category.name} - {os.path.basename(self.image.name)}"

class TrainedModel(models.Model):
    name = models.CharField(max_length=100)
    file_path = models.CharField(max_length=255)
    accuracy = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} (Accuracy: {self.accuracy:.2f})"