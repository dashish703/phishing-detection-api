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
from firewall import Firewall  # ✅ Custom module
from flask_sqlalchemy import SQLAlchemy
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

# --- OAuth2 Scopes ---
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# --- Database Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///settings.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# --- Load Ensemble Model from GCS ---
bucket_name = "phishing-model-files"
blob_name = "ensemble_phishing_model.pkl"

def download_model_from_gcs(bucket_name, blob_name):
    try:
        storage_client = storage.Client()
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

# --- Initialize Firewall ---
firewall = Firewall()

# --- Load Ensemble Model ---
try:
    model_path = download_model_from_gcs(bucket_name, blob_name)
    model = joblib.load(model_path)
except Exception as e:
    raise RuntimeError(f"Error loading the model: {e}")

# --- User Settings Model ---
class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), unique=True, nullable=False)
    notifications = db.Column(db.JSON, nullable=False)
    whitelist = db.Column(db.JSON, nullable=False)
    blacklist = db.Column(db.JSON, nullable=False)

    def __init__(self, user_id, notifications, whitelist, blacklist):
        self.user_id = user_id
        self.notifications = notifications
        self.whitelist = whitelist
        self.blacklist = blacklist

# --- Initialize DB ---
with app.app_context():
    db.create_all()

# --- Dashboard Stats ---
@app.route('/dashboard', methods=['GET'])
def get_dashboard_stats():
    try:
        stats = get_email_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_email_statistics():
    return {
        'totalScanned': 1247,
        'phishingEmails': 32,
        'suspiciousEmails': 18,
        'safeEmails': 1197
    }

# --- Update User Settings ---
@app.route("/updateSettings", methods=["POST"])
def update_settings():
    try:
        data = request.get_json()

        user_id = data.get("user_id")
        notifications = data.get("notifications")
        whitelist = data.get("whitelist")
        blacklist = data.get("blacklist")

        if not user_id or not notifications or not whitelist or not blacklist:
            return jsonify({"error": "Missing fields"}), 400

        user_settings = UserSettings.query.filter_by(user_id=user_id).first()

        if user_settings:
            user_settings.notifications = notifications
            user_settings.whitelist = whitelist
            user_settings.blacklist = blacklist
        else:
            user_settings = UserSettings(user_id, notifications, whitelist, blacklist)

        db.session.add(user_settings)
        db.session.commit()

        return jsonify({"message": "Settings updated successfully!"}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- Predict Endpoint ---
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("text", "")
        url = data.get("url", "")

        if not text or not url:
            return jsonify({"error": "Both 'text' and 'url' must be provided."}), 400

        return jsonify(predict_email(text, url))

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

def predict_email(text, url):
    text_embedding = get_bert_embedding(text)
    url_features = extract_url_features(url)
    combined = np.hstack([text_embedding, url_features]).reshape(1, -1)

    prediction = model.predict(combined)[0]
    confidence = max(model.predict_proba(combined)[0])  # confidence score

    if prediction == 1:
        firewall.log_phishing_attempt(text, url, prediction)

    return {"phishing": bool(prediction), "confidence": float(confidence)}

# --- Scan Gmail Inbox ---
@app.route('/scan-inbox', methods=['GET'])
def scan_inbox():
    try:
        flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
        service = build('gmail', 'v1', credentials=creds)

        results = service.users().messages().list(userId='me', maxResults=5).execute()
        messages = results.get('messages', [])

        scan_results = []

        for msg in messages:
            detail = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            snippet = detail.get('snippet', '')
            payload = {"text": snippet, "url": ""}

            result = predict_email(payload['text'], payload['url'])

            subject = next((h['value'] for h in detail.get('payload', {}).get('headers', []) if h['name'] == 'Subject'), "No Subject")

            scan_results.append({
                "subject": subject,
                "snippet": snippet,
                "phishing": result['phishing'],
                "confidence": result['confidence']
            })

        return jsonify({"emails": scan_results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- Reset Password Endpoint ---
@app.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        data = request.get_json()
        email = data.get("email", "").strip()

        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify({"error": "Valid email is required"}), 400

        # Placeholder logic
        print(f"[INFO] Reset link would be sent to {email}")

        return jsonify({"message": "If the email exists, a reset link has been sent."}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- Logs Retrieval ---
@app.route("/logs", methods=["GET"])
def get_logs():
    try:
        logs = firewall.get_recent_attempts()
        return jsonify({"logs": logs})
    except Exception as e:
        return jsonify({"error": f"Could not retrieve logs: {str(e)}"}), 500

# --- BERT Embedding ---
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

# --- URL Feature Extraction ---
def extract_url_features(url):
    parsed = urlparse(url)
    return np.array([
        len(url),
        len(parsed.netloc),
        len(parsed.path),
        int(bool(re.search(r"\d", url))),
        int(url.count('-')),
    ])

# --- Main ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
