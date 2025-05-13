import os
import re
import joblib
import numpy as np
import torch
import tempfile
from flask import Flask, request, jsonify
from flask_cors import CORS
from urllib.parse import urlparse
from transformers import DistilBertTokenizer, DistilBertModel
from google.cloud import storage
from firewall import Firewall  # ✅ Your custom module


# --- GCS Model Downloader ---
def download_model_from_gcs(bucket_name, blob_name):
    try:
        storage_client = storage.Client()  # Auth handled by GCP service account
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)

        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        blob.download_to_filename(tmp_file.name)
        return tmp_file.name
    except Exception as e:
        raise RuntimeError(f"Error downloading model from GCS: {e}")


# --- Initialize BERT ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

# --- Load Ensemble Model from GCS ---
bucket_name = "phishing-model-files"
blob_name = "ensemble_phishing_model.pkl"

try:
    model_path = download_model_from_gcs(bucket_name, blob_name)
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# ✅ Initialize Firewall
firewall = Firewall()


# --- Fetch Email Stats ---
@app.route('/dashboard', methods=['GET'])
def get_dashboard_stats():
    try:
        stats = get_email_statistics()  # Fetch statistics from your DB or other source
        return jsonify({
            'totalScanned': stats['totalScanned'],
            'phishingEmails': stats['phishingEmails'],
            'suspiciousEmails': stats['suspiciousEmails'],
            'safeEmails': stats['safeEmails'],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Get Email Statistics (example) ---
def get_email_statistics():
    # Example stats, replace with actual logic to fetch data from your database
    return {
        'totalScanned': 1247,
        'phishingEmails': 32,
        'suspiciousEmails': 18,
        'safeEmails': 1197
    }


# --- Prediction Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        url = data.get("url", "")

        if not text or not url:
            return jsonify({"error": "Both 'text' and 'url' must be provided."}), 400

        # Process features
        text_embedding = get_bert_embedding(text)
        url_features = extract_url_features(url)

        combined = np.hstack([text_embedding, url_features]).reshape(1, -1)

        prediction = model.predict(combined)[0]

        if prediction == 1:
            firewall.log_phishing_attempt(text, url, prediction)

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        print(f"[Prediction Error] {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


# --- Logs Retrieval ---
@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        recent_logs = firewall.get_recent_attempts()
        return jsonify({"logs": recent_logs})
    except Exception as e:
        print(f"[Logs Retrieval Error] {e}")
        return jsonify({"error": f"Could not retrieve logs: {str(e)}"}), 500


# --- Run Flask ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))  # Use PORT from Cloud Run
    app.run(host="0.0.0.0", port=port)
