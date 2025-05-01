from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('AI_APP.urls')),  # Lien vers ton app
    path('forms/', include('AI_APP.urls')),  # Lien vers ton app
]


