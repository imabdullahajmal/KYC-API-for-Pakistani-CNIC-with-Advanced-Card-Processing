from django.urls import path

from .views import id_detector

urlpatterns = [
    path('detect/', id_detector, name='id_detector'),
]
