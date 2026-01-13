from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib, uvicorn
from pydantic import BaseModel
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="🛡️ Job Fraud Pro API")
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

app_state = {}
flagged_jobs = []
all_predictions = []

class Job(BaseModel):
    title: str
    description: str
    company_profile: str = ""

def load_models():
    if 'model' not in app_state:
        try:
            app_state['model'] = joblib.load('model.pkl')
            app_state['vectorizer'] = joblib.load('vectorizer.pkl')
            logger.info("✅ Production models loaded!")
        except FileNotFoundError:
            logger.warning("⚠️ Auto-creating model...")
            fake_jobs = ["urgent whatsapp hire no experience"] * 50
            real_jobs = ["Senior Python Developer Django AWS"] * 50
            all_jobs = fake_jobs + real_jobs
            y = np.array([1]*50 + [0]*50)
            
            app_state['vectorizer'] = TfidfVectorizer(max_features=2000)
            X = app_state['vectorizer'].fit_transform(all_jobs)
            app_state['model'] = RandomForestClassifier(n_estimators=200)
            app_state['model'].fit(X, y)
            logger.info("✅ Model ready!")

@app.get("/")
async def health():
    load_models()
    return {"status": "🚀 Job Fraud Pro API ALIVE", "predictions": len(all_predictions)}

@app.post("/predict")
async def predict(job: Job):
    load_models()
    text = f"{job.title} {job.description} {job.company_profile}".lower()
    features = app_state['vectorizer'].transform([text])
    pred = app_state['model'].predict(features)[0]
    prob = app_state['model'].predict_proba(features)[0]
    
    # ✅ FIXED: EXACT response your Streamlit expects
    response = {
        "fake": bool(pred == 1),           # Boolean for if/else in Streamlit
        "fake_prob": float(prob[1]),       # For metric display  
        "confidence": float(max(prob))     # For AI Confidence metric
    }
    
    # Store prediction for dashboard
    prediction = {
        "id": len(all_predictions) + 1,
        "title": job.title[:100],
        "company": job.company_profile or "Unknown",
        "fake": bool(pred == 1),
        "fake_prob": float(prob[1]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    all_predictions.append(prediction)
    
    if pred == 1:
        flagged_jobs.append(prediction)
    
    logger.info(f"🔍 Predicted: {'FAKE' if pred==1 else 'REAL'} ({prob[1]:.1%})")
    return response  # ✅ Streamlit Job Scanner expects THIS exact format

@app.get("/dashboard/stats")
async def dashboard_stats():
    df = pd.DataFrame(flagged_jobs[-100:])
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    stats = {
        "total_flagged": len(flagged_jobs),
        "unique_companies": df['company'].nunique() if len(df) > 0 else 0,
        "today_count": len(df[df['timestamp'].str.contains(today_str)]) if len(df) > 0 else 0,
        "avg_probability": df['fake_prob'].mean() if len(df) > 0 else 0
    }
    return stats

@app.get("/dashboard/jobs")
async def dashboard_jobs():
    return flagged_jobs[-50:]

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)