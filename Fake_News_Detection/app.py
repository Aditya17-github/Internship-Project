from flask import Flask, request, render_template
import pickle
import os

app = Flask(__name__)

# Safe model loading
model_path = "model/fake_news_model.pkl"
vectorizer_path = "model/vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("❌ Model or vectorizer file not found. Train the model first.")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        news = request.form["news"]
        news_vec = vectorizer.transform([news])
        result = model.predict(news_vec)[0]
        prediction = "REAL NEWS 🟢" if result == 1 else "FAKE NEWS 🔴"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
