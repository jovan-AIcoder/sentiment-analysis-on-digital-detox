# Train a BI-LSTM model for sentiment prediction
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
# Fix random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Load labeled dataset
CSV_FILENAME = 'datasets/comments_labeled.csv'

try:
    df = pd.read_csv(CSV_FILENAME)
    print(f"Successfully loaded '{CSV_FILENAME}'. Total records: {len(df)}")
except FileNotFoundError:
    print(f"Error: File '{CSV_FILENAME}' not found. Please verify the path.")
    exit()

# Basic text cleaning function
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text.strip()

df['cleaned_text'] = df['text'].apply(preprocess_text)

# Drop empty rows after cleaning
df = df[df['cleaned_text'] != ''].reset_index(drop=True)

X = df['cleaned_text'].astype(dtype=str).to_numpy()
y = df['label'].astype(np.float32).to_numpy()

# Check dataset size
if len(df) < 10:
    print("Warning: Dataset is too small for training a deep learning model.")

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
)

print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# Text Vectorization Layer (Tokenization & Padding)
MAX_VOCAB_SIZE = 5000  # Maximum vocabulary size
MAX_SEQUENCE_LENGTH = 50  # Maximum words per comment
EMBEDDING_DIM = 64

vectorize_layer = layers.TextVectorization(
    max_tokens=MAX_VOCAB_SIZE,
    output_mode='int',
    output_sequence_length=MAX_SEQUENCE_LENGTH
)

# Adapt vectorizer to training data
vectorize_layer.adapt(X_train)

# Build Bi-LSTM Model Architecture
model = models.Sequential([
    # Input Layer for Raw Strings
    layers.Input(shape=(1,), dtype=tf.string),
    
    # Text Vectorization Layer
    vectorize_layer,
    
    # Embedding Layer (Learns word representations from scratch)
    layers.Embedding(
        input_dim=MAX_VOCAB_SIZE,
        output_dim=EMBEDDING_DIM,
        mask_zero=True
    ),
    
    # Spatial Dropout to prevent overfitting
    layers.SpatialDropout1D(0.2),
    
    # Bidirectional LSTM Layer
    layers.Bidirectional(layers.LSTM(32, return_sequences=False)),
    
    # Dense Layers
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output probability for Class 1 (Pro)
])

# Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# Train the Model
EPOCHS = 15
BATCH_SIZE = 16

print("\nStarting model training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)

# Model Evaluation
print("\n" + "="*40)
print("          MODEL EVALUATION RESULTS          ")
print("="*40)

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Con/OOT (0)', 'Pro (1)']))

# Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Con/OOT (0)', 'Pro (1)'],
            yticklabels=['Con/OOT (0)', 'Pro (1)'])
plt.title('Confusion Matrix - Bi-LSTM Predictor', fontsize=12, weight='bold')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

# Save plot
plt.savefig('figures/bilstm_confusion_matrix.png', dpi=300)
plt.close()

# Save Trained Model
MODEL_SAVE_PATH = 'models/bilstm_model.keras'
model.save(MODEL_SAVE_PATH)
print(f"Model successfully saved to '{MODEL_SAVE_PATH}'")

# Inference Test Function
def predict_comment_sentiment(comment_text):
    cleaned = preprocess_text(comment_text)
    input_data = tf.constant([cleaned], dtype=tf.string)
    prob = model.predict(input_data, verbose=0)[0][0]
    predicted_class = 1 if prob >= 0.5 else 0
    class_label = "Pro (Accept Challenge)" if predicted_class == 1 else "Con / Skeptical / OOT"
    
    print(f"\n[Test Comment]: \"{comment_text}\"")
    print(f"[Probability Pro]: {prob:.4f}")
    print(f"[Prediction]: Class {predicted_class} -> {class_label}")

# Demo Inference
print("\nRunning test inference on sample inputs:")
predict_comment_sentiment("Bisa banget, udah biasa ga main sosmed seharian")
predict_comment_sentiment("Ga mungkin lah bro, kerjaan gue semua di tiktok dan ig")

# ROC plot
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#2ecc71', lw=2, label=f'Bi-LSTM (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, weight='bold', pad=12)
plt.legend(loc="lower right", fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('figures/roc_curve.png', dpi=300)
print('Done!')