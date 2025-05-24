import os
import re
import joblib
import numpy as np
import torch
import tempfile
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from urllib.parse import urlparse
from transformers import DistilBertTokenizer, DistilBertModel
from google.cloud import storage
from firewall import Firewall  # ✅ Custom module
from flask_sqlalchemy import SQLAlchemy
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

# --- OAuth2 Scopes ---
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# --- Database Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///settings.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.environ.get("FLASK_SECRET", "your-secret-key")
db = SQLAlchemy(app)

# --- Use GCS secret key for Storage Client ---
# Secret is mounted at /secrets/GCS_KEY - set this in your Cloud Run service
GCS_KEY_PATH = "/secrets/GCS_KEY"

def get_storage_client():
    if os.path.exists(GCS_KEY_PATH):
        return storage.Client.from_service_account_json(GCS_KEY_PATH)
    else:
        # fallback for local dev if key not mounted
        return storage.Client()

# --- Load Ensemble Model from GCS ---
bucket_name = "phishing-model-files"
blob_name = "ensemble_phishing_model.pkl"

def download_model_from_gcs(bucket_name, blob_name):
    try:
        storage_client = get_storage_client()
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

# --- DB Models ---
class UserSettings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), unique=True, nullable=False)
    notifications = db.Column(db.JSON, nullable=False)
    whitelist = db.Column(db.JSON, nullable=False)
    blacklist = db.Column(db.JSON, nullable=False)

class EmailScanResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), nullable=False)
    subject = db.Column(db.String(255))
    snippet = db.Column(db.Text)
    phishing = db.Column(db.Boolean)
    confidence = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=db.func.now())

class GmailToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), unique=True, nullable=False)
    token = db.Column(db.JSON, nullable=False)

# --- Initialize DB ---
with app.app_context():
    db.create_all()

# --- OAuth2 Authorization ---
@app.route('/authorize', methods=['GET'])
def authorize():
    user_id = request.args.get('user_id')
    flow = Flow.from_client_secrets_file('credentials.json', scopes=SCOPES, redirect_uri='https://yourdomain.com/oauth2callback')
    auth_url, _ = flow.authorization_url(prompt='consent', access_type='offline', include_granted_scopes='true', state=user_id)
    return redirect(auth_url)

@app.route('/oauth2callback')
def oauth2callback():
    user_id = request.args.get('state')
    flow = Flow.from_client_secrets_file('credentials.json', scopes=SCOPES, redirect_uri='https://yourdomain.com/oauth2callback')
    flow.fetch_token(authorization_response=request.url)

    creds = flow.credentials
    token_data = {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes
    }

    existing = GmailToken.query.filter_by(user_id=user_id).first()
    if existing:
        existing.token = token_data
    else:
        db.session.add(GmailToken(user_id=user_id, token=token_data))
    db.session.commit()
    return "Authorization complete. You can return to the app."

# --- Gmail Scan Inbox ---
@app.route('/scan-inbox', methods=['GET'])
def scan_inbox():
    user_id = request.args.get('user_id')
    record = GmailToken.query.filter_by(user_id=user_id).first()

    if not record:
        return jsonify({"error": "User not authorized with Gmail"}), 403

    creds = Credentials.from_authorized_user_info(record.token, scopes=SCOPES)
    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())

    service = build('gmail', 'v1', credentials=creds)

    results = service.users().messages().list(userId='me', maxResults=5).execute()
    messages = results.get('messages', [])

    scan_results = []
    for msg in messages:
        detail = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
        snippet = detail.get('snippet', '')
        subject = next((h['value'] for h in detail.get('payload', {}).get('headers', []) if h['name'] == 'Subject'), "No Subject")

        result = predict_email(snippet, "")

        scan_results.append({
            "subject": subject,
            "snippet": snippet,
            "phishing": result['phishing'],
            "confidence": result['confidence']
        })

        scan_entry = EmailScanResult(
            user_id=user_id,
            subject=subject,
            snippet=snippet,
            phishing=result['phishing'],
            confidence=result['confidence']
        )
        db.session.add(scan_entry)
        db.session.commit()

    return jsonify({"emails": scan_results})

# --- Predict Email ---
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
    confidence = max(model.predict_proba(combined)[0])

    if prediction == 1:
        firewall.log_phishing_attempt(text, url, prediction)

    return {"phishing": bool(prediction), "confidence": float(confidence)}

# --- Embedding & Feature Helpers ---
def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

def extract_url_features(url):
    parsed = urlparse(url)
    return np.array([
        len(url),
        len(parsed.netloc),
        len(parsed.path),
        int(bool(re.search(r"\d", url))),
        int(url.count('-')),
    ])

# --- Dashboard Stats ---
@app.route('/dashboard', methods=['GET'])
def get_dashboard_stats():
    try:
        stats = get_email_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_email_statistics():
    total = EmailScanResult.query.count()
    phishing = EmailScanResult.query.filter_by(phishing=True).count()
    safe = EmailScanResult.query.filter_by(phishing=False).count()
    suspicious = EmailScanResult.query.filter(
        EmailScanResult.phishing == False,
        EmailScanResult.confidence < 0.7
    ).count()

    return {
        'totalScanned': total,
        'phishingEmails': phishing,
        'suspiciousEmails': suspicious,
        'safeEmails': safe - suspicious
    }

# --- User Settings ---
@app.route("/updateSettings", methods=["POST"])
def update_settings():
    try:
        data = request.get_json()
        user_id = data.get("user_id")
        notifications = data.get("notifications")
        whitelist = data.get("whitelist")
        blacklist = data.get("blacklist")

        if not user_id or notifications is None or whitelist is None or blacklist is None:
            return jsonify({"error": "Missing fields"}), 400

        user_settings = UserSettings.query.filter_by(user_id=user_id).first()

        if user_settings:
            user_settings.notifications = notifications
            user_settings.whitelist = whitelist
            user_settings.blacklist = blacklist
        else:
            user_settings = UserSettings(user_id=user_id, notifications=notifications, whitelist=whitelist, blacklist=blacklist)

        db.session.add(user_settings)
        db.session.commit()

        return jsonify({"message": "Settings updated successfully!"}), 200

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

# --- Password Reset (placeholder) ---
@app.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        data = request.get_json()
        email = data.get("email", "").strip()

        if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify({"error": "Valid email is required"}), 400

        print(f"[INFO] Reset link would be sent to {email}")
        return jsonify({"message": "If the email exists, a reset link has been sent."}), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500

# --- Main ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
