import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load datasets safely
fake = pd.read_csv(
    "data/Fake.csv",
    sep="\t",
    encoding="utf-8",
    engine="python",
    quoting=3,
    on_bad_lines="skip"
)

true = pd.read_csv(
    "data/True.csv",
    sep="\t",
    encoding="utf-8",
    engine="python",
    quoting=3,
    on_bad_lines="skip"
)

# Normalize columns
fake.columns = fake.columns.str.strip().str.lower()
true.columns = true.columns.str.strip().str.lower()

# Add labels
fake["label"] = 0
true["label"] = 1

# Combine
df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1).reset_index(drop=True)

# 🔥 TEXT CLEANING (CRITICAL FIX)
df["text"] = df["text"].astype(str)
df = df[df["text"].str.strip() != ""]
df = df[df["text"].str.lower() != "none"]
df = df.dropna(subset=["text"])

print("Dataset size after cleaning:", df.shape)

# Features & labels
X = df["text"]
y = df["label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorization
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# Evaluate
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model

os.makedirs("model", exist_ok=True)

pickle.dump(model, open("model/fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))

print("✅ Model & vectorizer saved successfully")

