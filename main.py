import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import re
from urllib.parse import urlparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset (assuming you have a CSV file with 'text' and 'label' columns)
# Label: 1 for phishing, 0 for legitimate
data = pd.read_csv('Phishing_data.json')

# Data Preprocessing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize text
    tokens = word_tokenize(text)
    # Remove stopwords
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing to the text column
data['cleaned_text'] = data['text'].apply(preprocess_text)

# Feature Engineering: Extract URL features
def extract_url_features(url):
    try:
        parsed_url = urlparse(url)
        features = {
            'url_length': len(url),
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_slashes': url.count('/'),
            'num_question_marks': url.count('?'),
            'num_equals': url.count('='),
            'num_at': url.count('@'),
            'domain_length': len(parsed_url.netloc),
            'path_length': len(parsed_url.path),
            'is_ip': 1 if re.match(r'^\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', parsed_url.netloc) else 0
        }
        return features
    except:
        return None

# Apply URL feature extraction (assuming 'url' column exists)
data['url_features'] = data['url'].apply(extract_url_features)

# Combine text and URL features
data['combined_features'] = data.apply(lambda row: row['cleaned_text'] + ' ' + str(row['url_features']), axis=1)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['combined_features'], data['label'], test_size=0.2, random_state=42)

# Text Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model Training: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# Model Evaluation
y_pred = model.predict(X_test_tfidf)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"Precision: {precision_score(y_test, y_pred)}")
print(f"Recall: {recall_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred)}")

# Save the model and vectorizer for future use
import joblib
joblib.dump(model, 'ensemble_phishing_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')