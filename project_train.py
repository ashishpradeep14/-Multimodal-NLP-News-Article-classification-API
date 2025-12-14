# project_train.py

import pandas as pd
import numpy as np
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# NLTK and Downloads
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    print("NLTK data (stopwords/punkt) not found. Downloading...")
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    STOPWORDS = set(stopwords.words('english'))
    print("NLTK data downloaded successfully.")

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Deep Learning Libraries
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# --- 1. Data Loading (SYNTHETIC DATA) ---
TARGET_CATEGORIES = ['POLITICS', 'WELLNESS', 'ENTERTAINMENT', 'BUSINESS', 'SPORTS', 'TRAVEL', 'TECH']

# Creating a larger synthetic dataset for better training simulation
data_synthetic = {
    'category': (['POLITICS'] * 2000) + (['SPORTS'] * 1500) + (['BUSINESS'] * 1200) + (['ENTERTAINMENT'] * 1800) + (['WELLNESS'] * 1000) + (['TRAVEL'] * 900) + (['TECH'] * 1600),
    'headline': [
        "President signs new budget bill for infrastructure funding", 
        "Senate clashes over health care reform legislation proposal", 
        "Global leaders meet for emergency climate summit in Europe",
        "Team wins championship with dramatic late-game goal scored",
        "Star player sets new league scoring record this season finale",
        "Controversial call mars otherwise clean cricket match victory",
        "Stock market closes high despite inflation fears for quarter",
        "Tech startup IPO sees massive early investor gains success",
        "Federal Reserve announces interest rate hike decision tomorrow",
        "A-list actor wins major award for blockbuster film release",
        "Reality TV star posts surprising reunion announcement on social",
        "Pop icon announces world tour dates for next year's schedule",
        "Simple tips for improving sleep and reducing stress daily",
        "New study links diet to long-term cognitive health benefits",
        "Experts debate best practices for mental health breaks at work",
        "Airlines prepare for busy holiday travel season rush volume",
        "Top destinations to visit on a budget this summer vacation",
        "New cruise line launches innovative environmental efforts plan",
        "New smartphone features revolutionary camera technology built",
        "Big data trends shaping the future of cloud computing usage",
        "AI model achieves human-level fluency in natural language speech"
    ] * 500 # Repeat headlines to simulate size
}

df_data = pd.DataFrame({
    'headline': data_synthetic['headline'][:len(data_synthetic['category'])],
    'category': data_synthetic['category']
})
df_data = df_data[df_data['category'].isin(TARGET_CATEGORIES)].copy()
print(f"Total samples for training: {len(df_data)}")

# --- 2. Preprocessing and Splitting ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 1]
    return " ".join(tokens)

df_data['cleaned_headline'] = df_data['headline'].apply(clean_text)

X = df_data['cleaned_headline']
Y = df_data['category']

# Encode target categories
label_encoder = LabelEncoder()
Y_encoded = label_encoder.fit_transform(Y)
NUM_CLASSES = len(label_encoder.classes_)
CLASS_NAMES = label_encoder.classes_

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded
)

# --- 3. Model 1: TF-IDF + Logistic Regression (Evaluation Only) ---
MAX_DF_FEATURES = 2000 
tfidf_vectorizer = TfidfVectorizer(max_features=MAX_DF_FEATURES)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

lr_model = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='saga', random_state=42)
lr_model.fit(X_train_tfidf, Y_train)
Y_pred_lr = lr_model.predict(X_test_tfidf)
print("\n--- Logistic Regression Accuracy ---")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred_lr):.4f}")

# --- 4. Model 2: LSTM (Deployment Model) ---
VOCAB_SIZE = 3000
MAX_LEN = 30
EMBEDDING_DIM = 100
LSTM_UNITS = 128
DROPOUT_RATE = 0.2
BATCH_SIZE = 64
EPOCHS = 5 # Reduced for quick run

# Tokenization and Padding
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)
X_train_padded = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post', truncating='post')
X_test_padded = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN, padding='post', truncating='post')
Y_train_ohe = to_categorical(Y_train, num_classes=NUM_CLASSES)
Y_test_ohe = to_categorical(Y_test, num_classes=NUM_CLASSES)

# Build LSTM Model
model_lstm = Sequential()
model_lstm.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_LEN, trainable=True))
model_lstm.add(SpatialDropout1D(DROPOUT_RATE))
model_lstm.add(LSTM(LSTM_UNITS, dropout=DROPOUT_RATE, recurrent_dropout=DROPOUT_RATE))
model_lstm.add(Dense(NUM_CLASSES, activation='softmax'))

model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTraining LSTM Model...")
model_lstm.fit(
    X_train_padded, 
    Y_train_ohe, 
    epochs=EPOCHS, 
    batch_size=BATCH_SIZE, 
    validation_data=(X_test_padded, Y_test_ohe), 
    verbose=0
)

Y_pred_lstm_ohe = model_lstm.predict(X_test_padded)
Y_pred_lstm = np.argmax(Y_pred_lstm_ohe, axis=1)

print("\n--- LSTM Model Accuracy ---")
print(f"Accuracy: {accuracy_score(Y_test, Y_pred_lstm):.4f}")
print("--- LSTM Classification Report ---")
print(classification_report(Y_test, Y_pred_lstm, target_names=CLASS_NAMES))


# --- 5. SAVING DEPLOYMENT ARTIFACTS ---

# 1. Save the Keras LSTM Model
# Saving with include_optimizer=False is often recommended for deployment
save_model(model_lstm, 'lstm_model.h5', include_optimizer=False) 

# 2. Save the Keras Tokenizer
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# 3. Save the Label Encoder
with open('label_encoder.pkl', 'wb') as handle:
    pickle.dump(label_encoder, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("\n✅ Deployment artifacts (lstm_model.h5, tokenizer.pkl, label_encoder.pkl) saved.")