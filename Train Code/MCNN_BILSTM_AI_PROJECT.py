import numpy as np
import pandas as pd
import fasttext
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Dense, LSTM, Bidirectional, Dropout, LayerNormalization, GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import re
from bs4 import BeautifulSoup

# --- Nettoyage du texte ---
def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@\w+|\#", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

# --- Chargement et préparation des données ---
df = pd.read_csv("IMDB Dataset.csv", encoding="utf-8")
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
df = df.drop_duplicates().reset_index(drop=True)
df['clean_text'] = df['review'].apply(clean_text)

# --- Chargement du modèle FastText pré-entraîné ---
ft = fasttext.load_model("cc.en.300.bin")

def sentence_to_vector(sentence, model, max_len=200):
    words = sentence.split()
    vectors = [model.get_word_vector(word) for word in words[:max_len]]
    if len(vectors) < max_len:
        vectors += [np.zeros(300)] * (max_len - len(vectors))
    return np.array(vectors)

print("Vectorisation des textes...")
X = np.array([sentence_to_vector(text, ft) for text in df["clean_text"]])
y = df["sentiment"].values

# --- Définition du modèle ---
def build_model():
    input_layer = Input(shape=(200, 300))
    conv1 = Conv1D(128, 3, activation='relu', padding='same')(input_layer)
    conv2 = Conv1D(128, 5, activation='relu', padding='same')(input_layer)
    merged = tf.keras.layers.concatenate([conv1, conv2])
    lstm_out = Bidirectional(LSTM(128, return_sequences=True))(merged)
    x = LayerNormalization()(lstm_out)
    x = GlobalMaxPooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_layer, outputs=output)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), tf.keras.metrics.AUC(name='prc', curve='PR')]
    )
    return model

# --- Validation croisée stratifiée ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_no = 1
histories = []
all_y_true = []
all_y_pred_prob = []

for train_index, test_index in skf.split(X, y):
    print(f'\n--- Training fold {fold_no} ---')
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    model = build_model()

    callbacks = [
        EarlyStopping(monitor='val_prc', patience=3, mode='max', restore_best_weights=True),
        ModelCheckpoint(f'best_model_fold_{fold_no}.keras', monitor='val_prc', save_best_only=True, mode='max', verbose=1)
    ]

    history = model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=256,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=1
    )

    histories.append(history)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print(f'Classification report for fold {fold_no}:')
    print(classification_report(y_test, y_pred))
    print(f'F1 score for fold {fold_no}: {f1_score(y_test, y_pred):.4f}')
    print(f'ROC AUC for fold {fold_no}: {roc_auc_score(y_test, y_pred_prob):.4f}')

    # Affichage matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Confusion Matrix - Fold {fold_no}')
    plt.xlabel('Prédictions')
    plt.ylabel('Vérités terrain')
    plt.show()

    all_y_true.extend(y_test)
    all_y_pred_prob.extend(y_pred_prob.flatten())

    fold_no += 1

# --- Évaluation globale ---
all_y_pred = (np.array(all_y_pred_prob) > 0.5).astype(int)
print('\n=== Overall classification report ===')
print(classification_report(all_y_true, all_y_pred))
print(f'Overall F1 score: {f1_score(all_y_true, all_y_pred):.4f}')
print(f'Overall ROC AUC: {roc_auc_score(all_y_true, all_y_pred_prob):.4f}')

# Matrice de confusion globale
cm_global = confusion_matrix(all_y_true, all_y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm_global, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Global')
plt.xlabel('Prédictions')
plt.ylabel('Vérités terrain')
plt.show()

# --- Visualisation de l'historique du dernier fold ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(histories[-1].history['accuracy'], label='Train Accuracy')
plt.plot(histories[-1].history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over epochs (last fold)')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(histories[-1].history['loss'], label='Train Loss')
plt.plot(histories[-1].history['val_loss'], label='Validation Loss')
plt.title('Loss over epochs (last fold)')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
