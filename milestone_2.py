# Milestone 2 Code: Model Training (Weeks 2-4) - TFIDF FIXED VERSION
# Logistic Regression and Random Forest with TF-IDF

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('stopwords', quiet=True)

# Create dataset from your provided sample
data = {
    'job_id': [1, 2, 3, 4, 5],
    'title': ['Software Engineer', 'Data Analyst', 'Work From Home Job', 'HR Executive', 'Online Survey Job'],
    'location': ['New York', 'Bangalore', 'Remote', 'Delhi', 'Remote'],
    'department': ['IT', 'Analytics', np.nan, 'HR', np.nan],
    'salary_range': ['60000-80000', '50000-70000', '100000+', '30000-40000', 'High income guaranteed'],
    'company_profile': ['Good tech company', 'Startup company', 'No company info', 'Well known firm', 'Unknown'],
    'description': ['Develop software applications', 'Analyze company data', 'Easy money from home', 
                   'Handle recruitment activities', 'Pay fees and earn money'],
    'requirements': ['Python Java SQL', 'Excel Python Statistics', 'No skills required', 
                    'Communication skills', 'No experience needed'],
    'benefits': ['Health insurance', 'Flexible hours', np.nan, 'PF ESI', np.nan],
    'telecommuting': [0, 0, 1, 0, 1],
    'has_company_logo': [1, 1, 0, 1, 0],
    'has_questions': [1, 0, 0, 1, 0],
    'fraudulent': [0, 0, 1, 0, 1]
}

df = pd.DataFrame(data)
print("Dataset created!")
print("Shape:", df.shape)
print("Fraud distribution:", df['fraudulent'].value_counts().to_dict())

# Combine text columns
text_columns = ['title', 'location', 'department', 'salary_range', 'company_profile', 
                'description', 'requirements', 'benefits']

df['text'] = df[text_columns].fillna('').astype(str).apply(
    lambda x: ' '.join(x.str.lower()), axis=1
)

# Target
X = df['text']
y = df['fraudulent']

# Preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Zs]', '', str(text))
    stop_words = set(stopwords.words('english'))
    ps = PorterStemmer()
    words = [ps.stem(w) for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)

X_preprocessed = X.apply(preprocess_text)

# Train-test split (small dataset)
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.4, random_state=42  # 40% test for demo
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# FIXED TF-IDF Vectorizer - NO random_state parameter
print("✅ TF-IDF Vectorization (Fixed)...")
vectorizer = TfidfVectorizer(
    max_features=50,      # Small for demo
    ngram_range=(1,2), 
    min_df=1,
    max_df=1.0
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("TF-IDF shapes:", X_train_vec.shape, X_test_vec.shape)

# LOGISTIC REGRESSION (Weeks 2-3)
print("" + "="*60)
print("LOGISTIC REGRESSION - MILESTONE 2")
print("="*60)

lr_model = LogisticRegression(
    class_weight='balanced', 
    max_iter=2000,
    random_state=42
)
lr_model.fit(X_train_vec, y_train)

y_pred_lr = lr_model.predict(X_test_vec)
lr_f1 = f1_score(y_test, y_pred_lr, zero_division=0)
print(f"✅ LR Test F1-Score: {lr_f1:.4f}")
print("LR Classification Report:")
print(classification_report(y_test, y_pred_lr, zero_division=0))

# Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
plt.figure(figsize=(6,4))
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.savefig('lr_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# RANDOM FOREST (Weeks 2-3) 
print("" + "="*60)
print("RANDOM FOREST BASELINE - MILESTONE 2")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=30,
    max_depth=3,
    class_weight='balanced',
    random_state=42
)
rf_model.fit(X_train_vec, y_train)

y_pred_rf = rf_model.predict(X_test_vec)
rf_f1 = f1_score(y_test, y_pred_rf, zero_division=0)
print(f"✅ RF Test F1-Score: {rf_f1:.4f}")
print("RF Classification Report:")
print(classification_report(y_test, y_pred_rf, zero_division=0))

# Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title('Random Forest Confusion Matrix')
plt.ylabel('Actual'); plt.xlabel('Predicted')
plt.savefig('rf_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Model Comparison Table (Milestone 2 requirement)
comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Test_F1_Score': [lr_f1, rf_f1]
})
print("" + "="*60)
print("MODEL COMPARISON TABLE - MILESTONE 2")
print("="*60)
print(comparison.round(4))

# Save ALL models and artifacts
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
joblib.dump(lr_model, 'logistic_regression_model.pkl')
joblib.dump(rf_model, 'random_forest_model.pkl')
comparison.to_csv('model_comparison.csv', index=False)

print("✅ MILESTONE 2 DELIVERABLES GENERATED:")
print("   • logistic_regression_model.pkl")
print("   • random_forest_model.pkl") 
print("   • tfidf_vectorizer.pkl")
print("   • lr_confusion_matrix.png")
print("   • rf_confusion_matrix.png")
print("   • model_comparison.csv")
print("🎉 MILESTONE 2 COMPLETE - Ready for Week 4!")