import os
import re
import json
import joblib
import numpy as np
import torch
import tempfile
import logging
import threading
from functools import wraps
from datetime import datetime, timedelta
from pathlib import Path

from flask import Flask, request, jsonify, redirect
from flask_cors import CORS
from transformers import DistilBertTokenizer, DistilBertModel
from google.cloud import storage, secretmanager
from firewall import Firewall
from flask_sqlalchemy import SQLAlchemy
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import jwt
from pyfcm import FCMNotification

# --- Flask Setup ---
app = Flask(__name__)
# Configure CORS to allow frontend origins (update with your frontend domain)
CORS(app, resources={r"/*": {"origins": ["https://your-frontend-domain.com", "http://localhost:8080"]}})

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
    logger.error("FLASK_SECRET environment variable is not set!")
    raise ValueError("FLASK_SECRET must be set")
app.secret_key = FLASK_SECRET

JWT_SECRET = os.environ.get("JWT_SECRET")
if not JWT_SECRET:
    logger.error("JWT_SECRET environment variable is not set!")
    raise ValueError("JWT_SECRET must be set")

JWT_ALGORITHM = 'HS256'
JWT_EXP_DELTA_SECONDS = 3600

FCM_SERVER_KEY = os.environ.get("FCM_SERVER_KEY")
if not FCM_SERVER_KEY:
    logger.warning("FCM_SERVER_KEY not set, push notifications disabled")

# --- Secret Manager ---
secret_client = secretmanager.SecretManagerServiceClient()
SECRET_NAME = "projects/61828726396/secrets/gmail-oauth-credentials/versions/latest"

def get_gmail_credentials():
    try:
        response = secret_client.access_secret_version(name=SECRET_NAME)
        secret_payload = response.payload.data.decode("UTF-8")
        credentials = json.loads(secret_payload)
        return credentials
    except Exception as e:
        logger.error(f"Failed to retrieve Gmail OAuth credentials: {e}")
        raise

# --- DB Init ---
db = SQLAlchemy(app)

# --- Google Cloud Storage ---
def get_storage_client():
    return storage.Client()

bucket_name = "phishing-model-files"
blob_name = "ensemble_phishing_model.pkl"

def download_model_from_gcs(bucket, blob):
    try:
        client = get_storage_client()
        blob_obj = client.bucket(bucket).blob(blob)
        tmp_file = tempfile.NamedTemporaryFile(delete=False)
        blob_obj.download_to_filename(tmp_file.name)
        logger.info(f"Model downloaded from GCS: {tmp_file.name}")
        return tmp_file.name
    except Exception as e:
        logger.error(f"Error downloading model from GCS: {e}")
        raise

# --- BERT Initialization ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
model_lock = threading.Lock()

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
    notifications = db.Column(db.JSON, nullable=False, default={})
    whitelist = db.Column(db.JSON, nullable=False, default=[])
    blacklist = db.Column(db.JSON, nullable=False, default=[])

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

class Notification(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), nullable=False)
    message = db.Column(db.Text, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.now())
    read = db.Column(db.Boolean, default=False)  # Added read column

class FCMToken(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.String(80), unique=True, nullable=False)
    fcm_token = db.Column(db.String(255), nullable=False)

with app.app_context():
    db.create_all()

# --- FCM Push Notifications ---
push_service = FCMNotification(api_key=FCM_SERVER_KEY) if FCM_SERVER_KEY else None

def send_push_notification(user_id, title, body, data):
    if not push_service:
        logger.warning("Push notifications disabled: FCM_SERVER_KEY not set")
        return
    fcm_token = FCMToken.query.filter_by(user_id=user_id).first()
    if not fcm_token:
        logger.warning(f"No FCM token for user_id: {user_id}")
        return
    try:
        result = push_service.notify_single_device(
            registration_id=fcm_token.fcm_token,
            message_title=title,
            message_body=body,
            data_message=data
        )
        logger.info(f"Push notification sent to user_id {user_id}: {result}")
    except Exception as e:
        logger.error(f"Failed to send push notification to user_id {user_id}: {e}")

# --- Auth Helpers ---
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

# --- OAuth Routes ---
REDIRECT_URI = os.getenv("OAUTH_REDIRECT_URI", "https://phishing-backend-61828726396.us-west1.run.app/oauth2callback")

@app.route('/authorize')
def authorize():
    user_id = request.args.get('user_id')
    if not user_id:
        logger.error("Missing user_id in /authorize")
        return jsonify({"error": "Missing user_id"}), 400
    try:
        credentials_info = get_gmail_credentials()
        flow = Flow.from_client_config(
            credentials_info,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        auth_url, _ = flow.authorization_url(
            prompt='consent',
            access_type='offline',
            include_granted_scopes='true',
            state=user_id
        )
        logger.info(f"OAuth authorization URL generated for user_id: {user_id}")
        return redirect(auth_url)
    except Exception as e:
        logger.error(f"OAuth flow init failed: {e}")
        return jsonify({"error": "OAuth init failed"}), 500

@app.route('/oauth2callback')
def oauth2callback():
    user_id = request.args.get('state')
    logger.info(f"OAuth callback received for user_id: {user_id}")
    if not user_id:
        logger.error("Missing state (user_id)")
        return "Missing state (user_id)", 400
    try:
        credentials_info = get_gmail_credentials()
        flow = Flow.from_client_config(
            credentials_info,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        flow.fetch_token(authorization_response=request.url)
        creds = flow.credentials
        logger.info(f"Access token received for user_id: {user_id}")
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
        logger.info(f"Gmail token saved for user_id: {user_id}")
        jwt_token = generate_jwt(user_id)
        return jsonify({"message": "Authorization complete.", "jwt_token": jwt_token})
    except Exception as e:
        logger.error(f"OAuth2 callback error for user_id {user_id}: {e}")
        return jsonify({"error": "OAuth2 callback failed"}), 400

# --- Gmail Service ---
def build_gmail_service(user_id):
    gmail_token = GmailToken.query.filter_by(user_id=user_id).first()
    if not gmail_token:
        logger.error(f"No Gmail OAuth2 token found for user_id: {user_id}")
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
            logger.info(f"Gmail token refreshed for user_id: {user_id}")
        except Exception as e:
            logger.error(f"Failed to refresh Gmail token for user_id {user_id}: {e}")
            raise
    try:
        service = build('gmail', 'v1', credentials=creds)
        logger.info(f"Gmail service built for user_id: {user_id}")
        return service
    except Exception as e:
        logger.error(f"Failed to build Gmail service for user_id {user_id}: {e}")
        raise

# --- BERT Prediction ---
def bert_encode(text):
    with model_lock:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
        outputs = bert_model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()
    return embedding

def predict_phishing(subject, snippet):
    combined = f"{subject} {snippet}"
    embedding = bert_encode(combined)
    prediction = model.predict(embedding)
    confidence = np.max(model.predict_proba(embedding))
    return bool(prediction[0]), confidence

# --- Prediction Endpoint ---
@app.route('/predict', methods=['POST'])
@require_auth
def predict(user_id):
    data = request.get_json()
    subject = data.get("subject", "")
    snippet = data.get("snippet", "")
    phishing, confidence = predict_phishing(subject, snippet)
    return jsonify({"phishing": phishing, "confidence": confidence})

# --- Gmail Scan Route ---
@app.route('/scan_gmail')
@require_auth
def scan_gmail(user_id):
    try:
        gmail_token = GmailToken.query.filter_by(user_id=user_id).first()
        if not gmail_token:
            logger.error(f"No Gmail token found for user_id: {user_id}")
            return jsonify({"error": "No Gmail OAuth token found"}), 404
        service = build_gmail_service(user_id)
        results = service.users().messages().list(userId='me', maxResults=10).execute()
        messages = results.get('messages', [])
        scan_results = []
        for msg in messages:
            try:
                msg_data = service.users().messages().get(
                    userId='me',
                    id=msg['id'],
                    format='metadata',
                    metadataHeaders=['Subject']
                ).execute()
                headers = msg_data.get('payload', {}).get('headers', [])
                subject = next((h['value'] for h in headers if h['name'] == 'Subject'), "")
                snippet = msg_data.get('snippet', "")
                phishing, confidence = predict_phishing(subject, snippet)
                if phishing:
                    send_push_notification(
                        user_id=user_id,
                        title="Phishing Email Detected",
                        body=f"Subject: {subject}",
                        data={"emailAddress": "unknown", "emailContent": snippet}
                    )
                result = EmailScanResult(
                    user_id=user_id,
                    subject=subject,
                    snippet=snippet,
                    phishing=phishing,
                    confidence=confidence
                )
                db.session.add(result)
                scan_results.append({
                    "subject": subject,
                    "snippet": snippet,
                    "phishing": phishing,
                    "confidence": confidence
                })
            except Exception as e:
                logger.error(f"Failed to scan message {msg.get('id')} for user_id {user_id}: {e}")
        db.session.commit()
        return jsonify(scan_results)
    except Exception as e:
        logger.error(f"Failed to fetch Gmail for user_id {user_id}: {e}")
        return jsonify({"error": "Gmail fetch failed"}), 500

# --- Dashboard Stats Route ---
@app.route('/dashboard_stats')
@require_auth
def dashboard_stats(user_id):
    phishing_count = EmailScanResult.query.filter_by(user_id=user_id, phishing=True).count()
    total_count = EmailScanResult.query.filter_by(user_id=user_id).count()
    return jsonify({"user_id": user_id, "phishing_count": phishing_count, "total_count": total_count})

# --- Email List Management Routes ---
@app.route('/get_list', methods=['GET'])
@require_auth
def get_list(user_id):
    settings = UserSettings.query.filter_by(user_id=user_id).first()
    if not settings:
        settings = UserSettings(user_id=user_id, notifications={}, whitelist=[], blacklist=[])
        db.session.add(settings)
        db.session.commit()
    list_type = request.args.get('listType')
    if list_type == 'whitelist':
        return jsonify({"emails": settings.whitelist})
    elif list_type == 'blacklist':
        return jsonify({"emails": settings.blacklist})
    return jsonify({"error": "Invalid list type"}), 400

@app.route('/add_to_list', methods=['POST'])
@require_auth
def add_to_list(user_id):
    data = request.get_json()
    email = data.get('email')
    list_type = data.get('listType')
    if not email or not list_type:
        return jsonify({"error": "Missing email or listType"}), 400
    settings = UserSettings.query.filter_by(user_id=user_id).first()
    if not settings:
        settings = UserSettings(user_id=user_id, notifications={}, whitelist=[], blacklist=[])
        db.session.add(settings)
    if list_type == 'whitelist':
        settings.whitelist.append({"email": email})
    elif list_type == 'blacklist':
        settings.blacklist.append({"email": email})
    else:
        return jsonify({"error": "Invalid list type"}), 400
    db.session.commit()
    return jsonify({"status": "Email added"})

@app.route('/remove_from_list', methods=['POST'])
@require_auth
def remove_from_list(user_id):
    data = request.get_json()
    email = data.get('email')
    list_type = data.get('listType')
    if not email or not list_type:
        return jsonify({"error": "Missing email or listType"}), 400
    settings = UserSettings.query.filter_by(user_id=user_id).first()
    if not settings:
        return jsonify({"error": "User settings not found"}), 404
    if list_type == 'whitelist':
        settings.whitelist = [e for e in settings.whitelist if e['email'] != email]
    elif list_type == 'blacklist':
        settings.blacklist = [e for e in settings.blacklist if e['email'] != email]
    else:
        return jsonify({"error": "Invalid list type"}), 400
    db.session.commit()
    return jsonify({"status": "Email removed"})

# --- Notification Endpoints ---
@app.route('/get_notifications')
@require_auth
def get_notifications(user_id):
    notifs = Notification.query.filter_by(user_id=user_id).all()
    return jsonify([
        {
            "id": n.id,
            "message": n.message,
            "timestamp": n.timestamp.isoformat(),
            "read": n.read
        } for n in notifs
    ])

@app.route('/submit_notification', methods=['POST'])
@require_auth
def submit_notification(user_id):
    data = request.get_json()
    message = data.get("message")
    db.session.add(Notification(user_id=user_id, message=message))
    db.session.commit()
    return jsonify({"status": "Notification saved"})

@app.route('/delete_notifications', methods=['DELETE'])
@require_auth
def delete_notifications(user_id):
    Notification.query.filter_by(user_id=user_id).delete()
    db.session.commit()
    return jsonify({"status": "All notifications deleted"})

@app.route('/notifications/<int:id>/mark_read', methods=['POST'])
@require_auth
def mark_notification_read(user_id, id):
    notification = Notification.query.filter_by(id=id, user_id=user_id).first()
    if not notification:
        return jsonify({"error": "Notification not found"}), 404
    notification.read = True
    db.session.commit()
    return jsonify({"status": "Notification marked as read"})

@app.route('/notifications/mark_all_read', methods=['POST'])
@require_auth
def mark_all_notifications_read(user_id):
    notifications = Notification.query.filter_by(user_id=user_id).all()
    for notif in notifications:
        notif.read = True
    db.session.commit()
    return jsonify({"status": "All notifications marked as read"})

@app.route('/notifications/<int:id>', methods=['DELETE'])
@require_auth
def delete_notification(user_id, id):
    notification = Notification.query.filter_by(id=id, user_id=user_id).first()
    if not notification:
        return jsonify({"error": "Notification not found"}), 404
    db.session.delete(notification)
    db.session.commit()
    return jsonify({"status": "Notification deleted"})

# --- FCM Token Registration ---
@app.route('/register_fcm_token', methods=['POST'])
@require_auth
def register_fcm_token(user_id):
    data = request.get_json()
    fcm_token = data.get('fcm_token')
    if not fcm_token:
        return jsonify({"error": "Missing FCM token"}), 400
    existing = FCMToken.query.filter_by(user_id=user_id).first()
    if existing:
        existing.fcm_token = fcm_token
    else:
        db.session.add(FCMToken(user_id=user_id, fcm_token=fcm_token))
    db.session.commit()
    return jsonify({"status": "FCM token registered"})

# --- Firewall Test ---
@app.route('/test_firewall')
def test_firewall():
    test_ip = request.remote_addr or "unknown"
    return jsonify({"firewall": "blocked" if firewall.is_blocked(test_ip) else "allowed"})

# --- App Run ---
if __name__ == '__main__':
    app.run(debug=True)