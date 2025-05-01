import numpy as np
import fasttext
from tensorflow import keras

# Charger uniquement FastText anglais et modèle sentiment anglais
ft_model = fasttext.load_model("cc.en.300.bin")
sentiment_model = keras.models.load_model("best_model_fold_3.keras")

def sentence_to_vector(sentence, ft_model, max_len=200):
    words = sentence.split()
    vectors = [ft_model.get_word_vector(word) for word in words]

    if len(vectors) > max_len:
        vectors = vectors[:max_len]
    elif len(vectors) < max_len:
        padding = [np.zeros(300)] * (max_len - len(vectors))
        vectors.extend(padding)

    return np.array(vectors)

def predict_sentiment(text):
    # Encoder le texte
    vectorized_text = np.expand_dims(sentence_to_vector(text, ft_model), axis=0)

    # Prédire sentiment
    sentiment_pred = sentiment_model.predict(vectorized_text)[0][0]
    sentiment = "Positive" if sentiment_pred >= 0.5 else "Negative"
    confidence = sentiment_pred * 100 if sentiment_pred >= 0.5 else (1 - sentiment_pred) * 100

    # Retourner résultat
    result = {
        "comment": text,
        "sentiment": {
            "value": sentiment,
            "confidence": f"{confidence:.2f}%"
        }
    }

    return result

# Exemple d'utilisation :
if __name__ == "__main__":
    import sys
    import json

    text = sys.argv[1]
    output = predict_sentiment(text)
    print(json.dumps(output, ensure_ascii=False))
