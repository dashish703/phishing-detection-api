import os
import re
import joblib
import numpy as np
import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import urlparse
from transformers import DistilBertTokenizer, DistilBertModel

# Load BERT tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

# Flask app setup
app = Flask(__name__)
CORS(app)

# Load ensemble model
model_path = os.path.join(os.path.dirname(__file__), "ensemble_phishing_model.pkl")
model = joblib.load(model_path)

# --- Utility Functions ---
def get_bert_embedding(text):
    try:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length"
        ).to(device)

        with torch.no_grad():
            outputs = bert_model(**inputs)

        return outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return np.zeros(768)

def extract_url_features(url):
    try:
        parsed = urlparse(url)
        return [
            len(url),
            len(parsed.netloc),
            url.count('.'),
            url.count('-'),
            url.count('/'),
            1 if re.match(r"\d+\.\d+\.\d+\.\d+", parsed.netloc) else 0
        ]
    except:
        return [0]*6

# --- Prediction Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        url = data.get("url", "")

        # Generate features
        text_embedding = get_bert_embedding(text)
        url_features = extract_url_features(url)

        # Combine features (774 total)
        combined = np.hstack([text_embedding, url_features]).reshape(1, -1)

        # Predict
        prediction = model.predict(combined)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- Run Server ---
if __name__ == "__main__":
    # Ensure Flask listens on all network interfaces and allows access from the Android emulator
    app.run(host="0.0.0.0", port=5000, debug=False)

