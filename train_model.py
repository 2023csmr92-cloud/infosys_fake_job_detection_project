import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Sample dataset (you can replace with CSV later)
data = {
    "text": [
        "Earn money quickly without work",
        "Looking for software engineer with experience",
        "Pay registration fee to get job",
        "Hiring Python developer for IT company"
    ],
    "label": [1, 0, 1, 0]  # 1 = Fake, 0 = Real
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(df["text"])

model = LogisticRegression()
model.fit(X, df["label"])

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))



print("Model trained and saved")