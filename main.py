# main.py
import pickle
import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 👇👇👇 CORRECTED IMPORT 👇👇👇
from tensorflow.keras.models import load_model 
# 👆👆👆 CORRECTED IMPORT 👆👆👆

from tensorflow.keras.preprocessing.sequence import pad_sequences 

# --- NLTK Configuration and Robust Download Check ---
try:
    STOPWORDS = set(stopwords.words('english'))
except LookupError:
    print("NLTK data (stopwords/punkt) not found. Attempting to download...")
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        STOPWORDS = set(stopwords.words('english'))
        print("NLTK data downloaded successfully.")
    except Exception as e:
        print(f"FATAL ERROR: Could not download NLTK data. Error: {e}")
        raise # Re-raise if essential data cannot be loaded

# --- Configuration & Artifacts Loading ---
MAX_LEN = 30 # Must match the length used during training

try:
    # 1. Load the Keras Model (Now load_model is defined)
    MODEL = load_model('lstm_model.h5', compile=False) # compile=False speeds up loading for deployment
    
    # 2. Load the Keras Tokenizer
    with open('tokenizer.pkl', 'rb') as f:
        TOKENIZER = pickle.load(f)
        
    # 3. Load the Label Encoder
    with open('label_encoder.pkl', 'rb') as f:
        LABEL_ENCODER = pickle.load(f)
        
    print("✅ All model artifacts loaded successfully.")
    
except FileNotFoundError:
    print("❌ ERROR: Model artifacts not found. Run 'project_train.py' first.")
    raise

# --- FastAPI Setup ---
app = FastAPI(title="News Classifier Deployment")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


# --- Preprocessing Function (Match Training) ---
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in STOPWORDS and len(word) > 1]
    return " ".join(tokens)

# --- Prediction Logic ---
def predict_headline_category(headline: str):
    # 1. Preprocess
    cleaned_text = clean_text(headline)
    
    # 2. Tokenize and Pad Sequence 
    sequence = TOKENIZER.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # 3. Predict
    prediction = MODEL.predict(padded_sequence)
    
    # 4. Decode Result
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_category = LABEL_ENCODER.inverse_transform([predicted_index])[0]
    confidence = prediction[0][predicted_index]
    
    return predicted_category, confidence


# --- FastAPI Routes ---

@app.get("/", name="home")
async def home(request: Request):
    """Serve the main classification HTML page."""
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict")
async def predict_route(request: Request, headline: str = Form(...)):
    """Handle the form submission and return the classification result."""
    
    category, confidence = predict_headline_category(headline)
    
    result_data = {
        "input_headline": headline,
        "predicted_category": category,
        "confidence": f"{confidence * 100:.2f}%"
    }
    
    return templates.TemplateResponse("index.html", {"request": request, "result": result_data})

# To run the app: uvicorn main:app --reload