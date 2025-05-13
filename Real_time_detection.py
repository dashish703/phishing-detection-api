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
from flask_sqlalchemy import SQLAlchemy  # New import for database

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

# --- Database Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///settings.db'  # You can change this to a production DB
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Load Ensemble Model from GCS ---
bucket_name = "phishing-model-files"
blob_name = "ensemble_phishing_model.pkl"

# Download model from GCS
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

# Initialize BERT
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

# --- Initialize Firewall ---
firewall = Firewall()

# --- Load Ensemble Model ---
try:
    model_path = download_model_from_gcs(bucket_name, blob_name)
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# --- User Settings Model (Database Schema) ---
class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), unique=True, nullable=False)
    notifications = db.Column(db.JSON, nullable=False)  # Store as JSON
    whitelist = db.Column(db.JSON, nullable=False)  # Store as JSON
    blacklist = db.Column(db.JSON, nullable=False)  # Store as JSON

    def __init__(self, user_id, notifications, whitelist, blacklist):
        self.user_id = user_id
        self.notifications = notifications
        self.whitelist = whitelist
        self.blacklist = blacklist

# Initialize the database (this is to create the tables)
with app.app_context():
    db.create_all()

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

# --- Update User Settings (New Endpoint) ---
@app.route("/updateSettings", methods=["POST"])
def update_settings():
    try:
        data = request.get_json()

        user_id = data.get("user_id")
        notifications = data.get("notifications")
        whitelist = data.get("whitelist")
        blacklist = data.get("blacklist")

        # Validate the input data
        if not user_id or not notifications or not whitelist or not blacklist:
            return jsonify({"error": "All fields ('user_id', 'notifications', 'whitelist', 'blacklist') must be provided."}), 400

        # Check if user settings already exist in the database
        user_settings = UserSettings.query.filter_by(user_id=user_id).first()

        if user_settings:
            # Update existing settings
            user_settings.notifications = notifications
            user_settings.whitelist = whitelist
            user_settings.blacklist = blacklist
        else:
            # Create new settings
            user_settings = UserSettings(
                user_id=user_id,
                notifications=notifications,
                whitelist=whitelist,
                blacklist=blacklist
            )

        # Commit changes to the database
        db.session.add(user_settings)
        db.session.commit()

        return jsonify({"message": "Settings updated successfully!"}), 200

    except Exception as e:
        print(f"[Settings Error] {e}")
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

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
