from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
import joblib
import os
from src.preprocess import clean_text

# Initialize App
app = FastAPI(title="Fake News Detector API")

# Load Models
MODEL_PATH = "models/"
try:
    model = joblib.load(os.path.join(MODEL_PATH, 'fake_news_model.pkl'))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, 'tfidf_vectorizer.pkl'))
except Exception as e:
    print(f"Error loading models: {e}")
    print("Did you run 'python -m src.train_model' first?")
    model = None
    vectorizer = None

# Define Request Structure
class NewsRequest(BaseModel):
    text: str
@app.get("/")
def home():
    return RedirectResponse(url="/docs")

@app.post("/predict")
def predict_news(request: NewsRequest):
    if not model or not vectorizer:
        raise HTTPException(status_code=500, detail="Model not loaded.")
    
    # 1. Clean
    cleaned_text = clean_text(request.text)
    
    # 2. Vectorize
    vectorized_text = vectorizer.transform([cleaned_text])
    
    # 3. Predict
    prediction = model.predict(vectorized_text)
    confidence = "High" # Simple placeholder, PAC doesn't output probability by default
    
    return {
        "prediction": prediction[0],
        "processed_text": cleaned_text
    }