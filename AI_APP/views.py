from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import subprocess
import json
from .predict import predict_sentiment  # Assurez-vous d'importer la fonction


def home(request):
    return render(request, 'home.html')



def formulaire_view(request):
    if request.method == 'POST':
        nom = request.POST.get('nom')
        email = request.POST.get('email')
        message = request.POST.get('message')
        print(nom, email, message)  # Tu peux traiter les données ici
    return render(request, 'formulaire.html')



@csrf_exempt  # temporaire si tu n'as pas encore géré CSRF côté frontend
def predict_view(request):
    if request.method == 'POST':
        try:
            # Charger les données JSON envoyées par le formulaire
            data = json.loads(request.body)
            comment = data.get('comment')

            if not comment:
                return JsonResponse({'error': 'Commentaire manquant'}, status=400)

            # Appeler la fonction de prédiction
            result = predict_sentiment(comment)

            # Retourner la réponse
            return JsonResponse(result)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Méthode non autorisée'}, status=405)