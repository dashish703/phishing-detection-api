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
from pathlib import Path

from flask import Flask, request, jsonify, redirect, abort
from flask_cors import CORS
from urllib.parse import urlparse
from transformers import DistilBertTokenizer, DistilBertModel
from google.cloud import storage, secretmanager
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
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///settings.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

FLASK_SECRET = os.environ.get("FLASK_SECRET")
if not FLASK_SECRET:
    logger.warning("FLASK_SECRET env var not set, using insecure default!")
app.secret_key = FLASK_SECRET or "your-default-insecure-secret"

JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    logger.warning("JWT_SECRET env var not set, using insecure default!")
JWT_SECRET = JWT_SECRET or "super-secret-jwt-key"

JWT_ALGORITHM = 'HS256'
JWT_EXP_DELTA_SECONDS = 3600  # 1 hour

db = SQLAlchemy(app)

# --- Google Cloud Storage ---
GCS_KEY_PATH = "/secrets/GCS_KEY"  # Mounted secret path on GCP Cloud Run
GMAIL_CREDENTIALS_PATH = "/secrets/GMAIL_CREDENTIALS"  # Mounted Gmail credentials JSON

def get_storage_client():
    if Path(GCS_KEY_PATH).exists():
        return storage.Client.from_service_account_json(GCS_KEY_PATH)
    return storage.Client()

# --- Secret Manager Client ---
secret_client = secretmanager.SecretManagerServiceClient()

def get_secret(secret_name: str, version: str = "latest") -> str:
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT environment variable not set")
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/{version}"
    response = secret_client.access_secret_version(name=secret_path)
    return response.payload.data.decode("UTF-8")

# --- Load client secrets file path for OAuth flow ---
CREDENTIALS_PATH = GMAIL_CREDENTIALS_PATH

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
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

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
        kwargs['user_id'] = user_id
        return f(*args, **kwargs)
    return decorated

# --- OAuth2 Endpoints ---
REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "https://phishing-backend-61828726396.us-west1.run.app/oauth2callback")

@app.route('/authorize')
def authorize():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "Missing user_id in query"}), 400

    try:
        flow = Flow.from_client_secrets_file(
            CREDENTIALS_PATH,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
    except Exception as e:
        logger.error(f"Failed to create OAuth Flow: {e}")
        return jsonify({"error": "OAuth flow initialization failed"}), 500

    auth_url, _ = flow.authorization_url(
        prompt='consent',
        access_type='offline',
        include_granted_scopes='true',
        state=user_id
    )
    return redirect(auth_url)

@app.route('/oauth2callback')
def oauth2callback():
    user_id = request.args.get('state')
    if not user_id:
        return "Missing state (user_id)", 400

    try:
        flow = Flow.from_client_secrets_file(
            CREDENTIALS_PATH,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        flow.fetch_token(authorization_response=request.url)
    except Exception as e:
        logger.error(f"OAuth callback failed: {e}")
        return jsonify({"error": "OAuth token fetch failed"}), 400

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
    return jsonify({"message": "Authorization complete. You can return to the app.", "jwt_token": jwt_token})

# --- Gmail API Utilities ---
def build_gmail_service(user_id):
    gmail_token = GmailToken.query.filter_by(user_id=user_id).first()
    if not gmail_token:
        raise ValueError("No Gmail OAuth2 token found for user")

    creds_data = gmail_token.token
    creds = Credentials(
        token=creds_data['token'],
        refresh_token=creds_data.get('refresh_token'),
        token_uri=creds_data['token_uri'],
        client_id=creds_data['client_id'],
        client_secret=creds_data['client_secret'],
        scopes=creds_data['scopes']
    )

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            gmail_token.token['token'] = creds.token
            db.session.commit()
        except Exception as e:
            logger.error(f"Failed to refresh Gmail token: {e}")
            raise

    return build('gmail', 'v1', credentials=creds)

# --- Phishing Prediction Helper ---
def bert_encode(text):
    with model_lock:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    return embedding

def predict_phishing(subject, snippet):
    combined_text = f"{subject} {snippet}"
    embedding = bert_encode(combined_text)
    prediction = model.predict(embedding)
    confidence = np.max(model.predict_proba(embedding))
    return bool(prediction[0]), confidence

# --- Routes ---
@app.route('/scan_gmail')
@require_auth
def scan_gmail(user_id):
    try:
        service = build_gmail_service(user_id)
        results = service.users().messages().list(userId='me', maxResults=10).execute()
        messages = results.get('messages', [])
    except Exception as e:
        logger.error(f"Failed to fetch Gmail messages for user {user_id}: {e}")
        return jsonify({"error": "Failed to fetch Gmail messages"}), 500

    scan_results = []
    for msg in messages:
        try:
            msg_data = service.users().messages().get(userId='me', id=msg['id'], format='metadata', metadataHeaders=['Subject']).execute()
            headers = msg_data.get('payload', {}).get('headers', [])
            subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "")
            snippet = msg_data.get('snippet', "")
            phishing, confidence = predict_phishing(subject, snippet)
            scan_results.append({"subject": subject, "snippet": snippet, "phishing": phishing, "confidence": confidence})
            db.session.add(EmailScanResult(user_id=user_id, subject=subject, snippet=snippet, phishing=phishing, confidence=confidence))
        except Exception as e:
            logger.warning(f"Failed to scan message id={msg['id']}: {e}")
            continue

    db.session.commit()
    return jsonify({"results": scan_results})

@app.route('/dashboard_stats')
@require_auth
def dashboard_stats(user_id):
    phishing_count = EmailScanResult.query.filter_by(user_id=user_id, phishing=True).count()
    total_count = EmailScanResult.query.filter_by(user_id=user_id).count()
    return jsonify({"user_id": user_id, "phishing_count": phishing_count, "total_count": total_count})

@app.route('/test_firewall')
def test_firewall():
    test_ip = request.remote_addr or "unknown"
    return jsonify({"firewall": "blocked" if firewall.is_blocked(test_ip) else "allowed"})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
