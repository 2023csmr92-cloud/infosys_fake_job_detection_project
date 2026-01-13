import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

print("🔬 **COMPLETE Model Evaluation (Self-Contained)**")

# === CREATE TEST DATA INSIDE SCRIPT ===
fake_jobs = [
    "urgent hire whatsapp only no experience easy money daily payment telegram now",
    "CEO salary 200k work 2 hours remote contact telegram whatsapp payment today",
    "data entry typing jobs from home 10k per hour no investment required"
] * 30

real_jobs = [
    "Senior Python Developer Django Flask PostgreSQL AWS 3+ years experience",
    "Data Scientist ML NLP PyTorch TensorFlow 5+ years PhD preferred",
    "Fullstack Engineer React Node.js MongoDB Docker Kubernetes"
] * 30

# Combine + labels
all_jobs = fake_jobs + real_jobs
y_true = np.array([1]*len(fake_jobs) + [0]*len(real_jobs))  # 1=Fake, 0=Real

# Vectorize
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(all_jobs)

# Load model or create one
try:
    model = joblib.load('model.pkl')
    print("✅ Loaded existing model")
except:
    print("⚠️ Creating demo model...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X, y_true)

# === SPLIT DATA ===
X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, random_state=42)

# === EVALUATE ===
y_pred = model.predict(X_test)
cv_scores = cross_val_score(model, X, y_true, cv=5)

# Results
results = {
    'accuracy': accuracy_score(y_test, y_pred),
    'cv_accuracy': cv_scores.mean(),
    'cv_std': cv_scores.std(),
    'precision_fake': classification_report(y_test, y_pred, output_dict=True)['1']['precision'],
    'recall_fake': classification_report(y_test, y_pred, output_dict=True)['1']['recall'],
    'f1_fake': classification_report(y_test, y_pred, output_dict=True)['1']['f1-score'],
    'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
}

print(f"✅ Test Accuracy: {results['accuracy']:.3f}")
print(f"✅ CV Accuracy: {results['cv_accuracy']:.3f} ± {results['cv_std']:.3f}")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))

# SAVE for Streamlit
joblib.dump(results, 'model_evaluation.pkl')
joblib.dump({'model': model, 'results': results}, 'evaluation_complete.pkl')
print("💾 evaluation_complete.pkl SAVED! 🎉")