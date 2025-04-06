from django import forms

class ImageUploadForm(forms.Form):
    image = forms.ImageField(widget=forms.FileInput(attrs={'class': 'form-control', 'id': 'upload-image'}))
    label = forms.CharField(max_length=100, widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': '라벨 이름 입력 (예: 고양이, 강아지)'}))

class TrainingForm(forms.Form):
    epochs = forms.IntegerField(min_value=1, max_value=100, initial=10, 
                               widget=forms.NumberInput(attrs={'class': 'form-control'}))
    batch_size = forms.IntegerField(min_value=1, max_value=256, initial=32,
                                   widget=forms.NumberInput(attrs={'class': 'form-control'}))
    learning_rate = forms.FloatField(min_value=0.0001, max_value=0.1, initial=0.001,
                                    widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.0001'}))
    validation_split = forms.FloatField(min_value=0.1, max_value=0.5, initial=0.2,
                                       widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.05'}))