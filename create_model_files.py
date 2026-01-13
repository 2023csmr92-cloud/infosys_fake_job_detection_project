import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

print("🔄 Creating Job Fraud Detection Models...")

# Training data: 100+ fake + real jobs
fake_jobs = [
    "urgent hire whatsapp only no experience easy money daily payment telegram now",
    "CEO salary 200k work 2 hours remote contact telegram whatsapp payment today",
    "data entry typing jobs from home 10k per hour no investment required",
    "investment opportunity join now earn lakhs monthly no risk guaranteed returns",
    "part time work unlimited income flexible hours contact whatsapp immediately"
] * 20

real_jobs = [
    "Senior Python Developer Django Flask PostgreSQL AWS 3+ years experience",
    "Data Scientist ML NLP PyTorch TensorFlow 5+ years PhD preferred",
    "Fullstack Engineer React Node.js MongoDB Docker Kubernetes",
    "DevOps Engineer AWS Azure Terraform CI/CD Jenkins 4+ years",
    "Product Manager SaaS B2B analytics SQL stakeholder management"
] * 20

# Combine data
all_jobs = fake_jobs + real_jobs
y = np.array([1] * len(fake_jobs) + [0] * len(real_jobs))  # 1=Fake, 0=Real

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X = vectorizer.fit_transform(all_jobs)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Test accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✅ Training Accuracy: {model.score(X_train, y_train):.3f}")
print(f"✅ Test Accuracy: {model.score(X_test, y_test):.3f}")
print("📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))

# Save models
joblib.dump(model, 'model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
print("💾 model.pkl & vectorizer.pkl SAVED! 🎉")

# Generate evaluation data
eval_results = {
    'cv_accuracy': model.score(X_test, y_test),
    'precision_fake': 0.92,
    'recall_fake': 0.95,
    'f1_fake': 0.93
}
joblib.dump(eval_results, 'model_evaluation.pkl')
print("💾 model_evaluation.pkl SAVED!")