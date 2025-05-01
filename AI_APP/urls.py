from django.urls import path
from . import views
from .views import formulaire_view, predict_view


urlpatterns = [
    path('', views.home, name='home'),
    path('forms/', formulaire_view, name='formulaire'),  # 🔥 Correction ici
    path('predict/', predict_view, name='predict'),
]


