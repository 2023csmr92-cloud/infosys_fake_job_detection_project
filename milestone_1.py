import numpy as np
import pandas as pd  # ADD THIS MISSING IMPORT
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')

# --- PART 1: DATA LOADING & CLEANING ---
# 1. Dataset cleaned and transformed
df = pd.read_csv('fake_job_postings.csv')  # FIXED: pd.read_csv instead of id.read_csv
df.fillna('', inplace=True)  # Handling missing values

def clean_text(text):
    # Removing HTML tags, special characters, and normalizing (lowercase/tokenize)
    text = re.sub(r'<.*?>', '', text) 
    text = text.lower() 
    text = re.sub(r'[^a-zA-Zs]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [w for w in tokens if w not in stop_words]
    return " ".join(cleaned)

# Create the cleaned text column
df['cleaned_text'] = (df['title'] + " " + df['description']).apply(clean_text)

# --- PART 2: EXPLORATORY DATA ANALYSIS (EDA) ---
# Statistical Analysis & Class Imbalance
print("--- Class Imbalance Analysis ---")
print(df['fraudulent'].value_counts(normalize=True))

# Visual Analysis: Distributions
plt.figure(figsize=(6,4))
sns.countplot(x='fraudulent', data=df)
plt.title('Class Distribution (0: Real, 1: Fake)')
plt.savefig('distribution_plot.png')
plt.show()

# Visual Analysis: Word Clouds (FIXED: moved plot show)
fake_text = " ".join(df[df['fraudulent'] == 1]['cleaned_text'])
wc = WordCloud(width=800, height=400, background_color='white').generate(fake_text)
plt.figure(figsize=(10,5))
plt.imshow(wc)
plt.axis('off')
plt.title('Patterns in Fake Jobs')
plt.savefig('wordcloud_fake.png')
plt.show()  # MOVED HERE to fix display issue

# --- PART 3: FEATURE EXTRACTION ---
y = df['fraudulent']
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['cleaned_text'])  # Feature matrix 

# Save files for Milestone 2
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')
joblib.dump(X, 'tfidf_features.pkl')
joblib.dump(y, 'labels.pkl')

print("--- Milestone 1 Complete ---")
print(f"Feature Matrix Shape: {X.shape}")