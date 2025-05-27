import os
import re
import joblib
import numpy as np
import torch
import tempfile
import logging
import threading
from functools import wraps
from datetime import datetime, timedelta

from flask import Flask, request, jsonify, redirect, abort
from flask_cors import CORS
from urllib.parse import urlparse
from transformers import DistilBertTokenizer, DistilBertModel
from google.cloud import storage
from firewall import Firewall
from flask_sqlalchemy import SQLAlchemy
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request

import jwt  # PyJWT for JWT token

# --- Flask Setup ---
app = Flask(__name__)
CORS(app)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# --- OAuth2 Scopes ---
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# --- Configurations ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///settings.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = os.environ.get("FLASK_SECRET", "your-secret-key")
JWT_SECRET = os.environ.get("JWT_SECRET", "super-secret-jwt-key")
JWT_ALGORITHM = 'HS256'
JWT_EXP_DELTA_SECONDS = 3600  # 1 hour

db = SQLAlchemy(app)

# --- Google Cloud Storage ---
GCS_KEY_PATH = "/secrets/GCS_KEY"  # Mounted secret path on GCP Cloud Run

def get_storage_client():
    if os.path.exists(GCS_KEY_PATH):
        return storage.Client.from_service_account_json(GCS_KEY_PATH)
    return storage.Client()

# --- Load Phishing Model from GCS ---
bucket_name = "phishing-model-files"
blob_name = "ensemble_phishing_model.pkl"

def download_model_from_gcs(bucket, blob):
    try:
        client = get_storage_client()
        blob_obj = client.bucket(bucket).blob(blob)
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        blob_obj.download_to_filename(tmp_file.name)
        logger.info(f"Model downloaded from GCS bucket '{bucket}' blob '{blob}' to '{tmp_file.name}'")
        return tmp_file.name
    except Exception as e:
        logger.error(f"Error downloading model from GCS: {e}")
        raise

# --- BERT Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
model_lock = threading.Lock()  # Lock for thread-safe inference

# --- Firewall ---
firewall = Firewall()

# --- Load Phishing Model ---
model_path = download_model_from_gcs(bucket_name, blob_name)
model = joblib.load(model_path)
logger.info("Phishing model loaded successfully.")

# --- Database Models ---
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

with app.app_context():
    db.create_all()

# --- Authentication Utilities ---

def generate_jwt(user_id):
    payload = {
        'user_id': user_id,
        'exp': datetime.utcnow() + timedelta(seconds=JWT_EXP_DELTA_SECONDS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token

def verify_jwt(token):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload['user_id']
    except (jwt.ExpiredSignatureError, jwt.InvalidTokenError) as e:
        logger.warning(f"JWT verification failed: {e}")
        return None

def require_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', None)
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"error": "Authorization header missing or invalid"}), 401
        token = auth_header.split(' ')[1]
        user_id = verify_jwt(token)
        if not user_id:
            return jsonify({"error": "Invalid or expired token"}), 401
        # Attach user_id to kwargs for route use
        kwargs['user_id'] = user_id
        return f(*args, **kwargs)
    return decorated

# --- OAuth2 Endpoints ---
REDIRECT_URI = "https://phishing-backend-61828726396.us-west1.run.app/oauth2callback"

@app.route('/authorize')
def authorize():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id in query"}), 400

    flow = Flow.from_client_secrets_file(
        'credentials.json',
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(
        prompt='consent',
        access_type='offline',
        include_granted_scopes='true',
        state=user_id
    )
    logger.info(f"Authorization URL generated for user_id={user_id}")
    return redirect(auth_url)

@app.route('/oauth2callback')
def oauth2callback():
    user_id = request.args.get('state')
    if not user_id:
        return "Missing state (user_id)", 400

    flow = Flow.from_client_secrets_file(
        'credentials.json',
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
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

    jwt_token = generate_jwt(user_id)
    logger.info(f"OAuth2 complete and token stored for user_id={user_id}")

    return jsonify({
        "message": "Authorization complete. You can return to the app.",
        "jwt_token": jwt_token
    })

# --- Gmail Scan Endpoint ---
@app.route('/scan-inbox')
@require_auth
def scan_inbox(user_id):
    record = GmailToken.query.filter_by(user_id=user_id).first()
    if not record:
        return jsonify({"error": "User not authorized or no token found"}), 403

    creds = Credentials(
        token=record.token['token'],
        refresh_token=record.token['refresh_token'],
        token_uri=record.token['token_uri'],
        client_id=record.token['client_id'],
        client_secret=record.token['client_secret'],
        scopes=record.token['scopes']
    )
    try:
        if not creds.valid and creds.refresh_token:
            creds.refresh(Request())
            record.token['token'] = creds.token
            db.session.commit()
            logger.info(f"Refreshed OAuth token for user_id={user_id}")
    except Exception as e:
        logger.error(f"Error refreshing token for user_id={user_id}: {e}")
        return jsonify({"error": "Failed to refresh OAuth token"}), 500

    try:
        service = build('gmail', 'v1', credentials=creds)
        messages_resp = service.users().messages().list(userId='me', maxResults=5).execute()
        messages = messages_resp.get('messages', [])
    except Exception as e:
        logger.error(f"Gmail API call failed for user_id={user_id}: {e}")
        return jsonify({"error": "Failed to fetch emails from Gmail"}), 500

    results = []

    for msg in messages:
        try:
            detail = service.users().messages().get(userId='me', id=msg['id'], format='full').execute()
            snippet = detail.get('snippet', '')
            subject = next(
                (h['value'] for h in detail.get('payload', {}).get('headers', []) if h['name'] == 'Subject'),
                "No Subject"
            )
            result = predict_email(snippet, "")
            results.append({
                "subject": subject,
                "snippet": snippet,
                "phishing": result['phishing'],
                "confidence": result['confidence']
            })
            db.session.add(EmailScanResult(
                user_id=user_id,
                subject=subject,
                snippet=snippet,
                phishing=result['phishing'],
                confidence=result['confidence']
            ))
            db.session.commit()
        except Exception as e:
            logger.error(f"Error processing message id={msg['id']} for user_id={user_id}: {e}")
            continue

    logger.info(f"Scanned inbox for user_id={user_id}, {len(results)} emails processed.")
    return jsonify({"emails": results})

# --- Phishing Prediction Logic ---
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    url = data.get("url", "")
    if not text or not url:
        return jsonify({"error": "Missing 'text' or 'url'"}), 400

    # Limit input length to prevent abuse
    if len(text) > 2000 or len(url) > 500:
        return jsonify({"error": "Input text or URL too long"}), 400

    try:
        result = predict_email(text, url)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

def predict_email(text, url):
    with model_lock:
        text_vec = get_bert_embedding(text)
        url_vec = extract_url_features(url)
        features = np.hstack([text_vec, url_vec]).reshape(1, -1)
        prediction = model.predict(features)[0]
        confidence = max(model.predict_proba(features)[0])
        if prediction == 1:
            firewall.log_phishing_attempt(text, url, prediction)
        return {"phishing": bool(prediction), "confidence": float(confidence)}

def get_bert_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
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
        int(url.count('-'))
    ])

# --- Dashboard Stats ---
@app.route('/dashboard')
@require_auth
def dashboard(user_id):
    total = EmailScanResult.query.filter_by(user_id=user_id).count()
    phishing = EmailScanResult.query.filter_by(user_id=user_id, phishing=True).count()
    suspicious = EmailScanResult.query.filter(
        EmailScanResult.user_id==user_id,
        EmailScanResult.phishing == False,
        EmailScanResult.confidence < 0.7
    ).count()
    safe = total - phishing - suspicious

    return jsonify({
        'totalScanned': total,
        'phishingEmails': phishing,
        'suspiciousEmails': suspicious,
        'safeEmails': safe
    })

# --- Settings Update ---
@app.route("/updateSettings", methods=["POST"])
@require_auth
def update_settings(user_id):
    data = request.get_json(force=True)
    notifications = data.get("notifications")
    whitelist = data.get("whitelist")
    blacklist = data.get("blacklist")
    if notifications is None or whitelist is None or blacklist is None:
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
    logger.info(f"Settings updated for user_id={user_id}")
    return jsonify({"message": "Settings updated successfully!"})

# --- Logs Retrieval ---
@app.route("/logs")
@require_auth
def logs(user_id):
    # Optionally filter logs per user here, currently global
    recent = firewall.get_recent_attempts()
    logger.info(f"Logs retrieved by user_id={user_id}")
    return jsonify({"logs": recent})

# --- Password Reset Placeholder ---
@app.route('/reset-password', methods=['POST'])
def reset_password():
    email = request.get_json(force=True).get("email", "").strip()
    if not email or not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return jsonify({"error": "Valid email required"}), 400
    logger.info(f"[INFO] Password reset link would be sent to {email}")
    return jsonify({"message": "If the email exists, a reset link has been sent."})

# --- App Runner ---
if __name__ == "__main__":
    # Cloud Run expects app on 0.0.0.0:8080
    app.run(host='0.0.0.0', port=8080)
