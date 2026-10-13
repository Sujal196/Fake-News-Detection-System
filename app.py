from flask import Flask, render_template, request, jsonify, send_from_directory

import os
import sys
from datetime import datetime, timedelta, timezone
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

import pymongo
from pymongo import MongoClient

from data_preprocessing import TextPreprocessor
from ml_models import FakeNewsDetector
from flask_cors import CORS
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)
CORS(app)

detector = None

preprocessor = None

# MongoDB Configuration
MONGO_URI = os.environ.get(
    'MONGODB_URI',
    'mongodb+srv://sujalsahu196_db_user:<db_password>@drishtix.dzxbcvj.mongodb.net/truthscan?retryWrites=true&w=majority&appName=DrishtiX'
)
MONGO_DB_NAME = os.environ.get('MONGODB_DB_NAME', 'truthscan')
MONGO_COLLECTION_NAME = os.environ.get('MONGODB_COLLECTION', 'history')

mongo_client = None
mongo_collection = None

def get_mongo_collection():
    global mongo_client, mongo_collection
    if mongo_collection is not None:
        return mongo_collection

    uri = os.environ.get('MONGODB_URI', MONGO_URI)
    if not uri or '<db_password>' in uri:
        return None

    try:
        if mongo_client is None:
            mongo_client = MongoClient(
                uri,
                serverSelectionTimeoutMS=4000,
                connectTimeoutMS=4000
            )
            # Verify connectivity
            mongo_client.admin.command('ping')
            print("[MongoDB] Successfully connected to MongoDB Atlas!")
        db = mongo_client[MONGO_DB_NAME]
        mongo_collection = db[MONGO_COLLECTION_NAME]
        return mongo_collection
    except Exception as e:
        print(f"[MongoDB] Connection error: {e}")
        mongo_client = None
        mongo_collection = None
        return None


def get_ist_time():
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist_tz).strftime('%Y-%m-%d %H:%M:%S')

def init_db():
    """Initialize MongoDB collection and indexes."""
    try:
        col = get_mongo_collection()
        if col is not None:
            col.create_index([("session_id", pymongo.ASCENDING)])
            col.create_index([("timestamp", pymongo.DESCENDING)])
            print("[MongoDB] Indexes initialized successfully.")
    except Exception as me:
        print(f"[MongoDB] Index init notice: {me}")

def initialize_model():
    global detector, preprocessor
    init_db()
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'models', 'final_model.pkl')
        vectorizer_path = os.path.join(base_dir, 'models', 'final_vectorizer.pkl')

        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            print(f"Loading model from: {model_path}")
            detector = FakeNewsDetector()
            detector.load_model(model_path, vectorizer_path)
        else:
            print(f"Model not found at {model_path}")
            return False

        preprocessor = TextPreprocessor()
        print("Model initialized successfully!")
        return True

    except Exception as e:
        print(f"Error initializing model: {str(e)}")
        import traceback
        traceback.print_exc()
        detector = None
        try:
            preprocessor = TextPreprocessor()
        except:
            preprocessor = None
        return False

def fallback_predict(text):
    text_lower = text.lower()
    fake_indicators = [
        'shocking', 'secret', 'miracle', 'reveals',
        'doctors hate', 'big pharma', 'conspiracy', 'alien',
        'overnight', 'instant', 'magical', 'cure all'
    ]
    real_indicators = [
        'study', 'research', 'scientists', 'published', 'peer-reviewed',
        'university', 'journal', 'clinical', 'trial', 'analysis', 'breakthrough'
    ]
    fake_score = sum(1 for word in fake_indicators if word in text_lower)
    real_score = sum(1 for word in real_indicators if word in text_lower)
    if fake_score > real_score:
        prediction = 'Fake News'
        confidence = min(0.9, 0.5 + fake_score * 0.1)
        probabilities = {
            'Fake News': min(0.9, 0.5 + fake_score * 0.1),
            'Real News': max(0.1, 0.5 - fake_score * 0.1)
        }
    else:
        prediction = 'Real News'
        confidence = min(0.9, 0.5 + real_score * 0.1)
        probabilities = {
            'Fake News': max(0.1, 0.5 - real_score * 0.1),
            'Real News': min(0.9, 0.5 + real_score * 0.1)
        }
    temp_detector = FakeNewsDetector()
    explanation = temp_detector.generate_explanation(text, None, None, 0 if prediction == 'Fake News' else 1)
    return {
        'prediction': prediction,
        'confidence': confidence,
        'probabilities': probabilities,
        'explanation': explanation
    }

def save_to_db(text, prediction, confidence, timestamp, session_id):
    """Save a prediction result to MongoDB Atlas."""
    try:
        col = get_mongo_collection()
        if col is not None:
            doc = {
                'input_text': text,
                'prediction': prediction,
                'confidence': float(confidence),
                'timestamp': timestamp,
                'session_id': session_id or ''
            }
            col.insert_one(doc)
            print(f"[MongoDB] Saved to Atlas: {text[:50]}... at {timestamp}")
        else:
            print(f"[MongoDB] Skipped saving: MongoDB is not connected yet (check .env / MONGODB_URI).")
    except Exception as m_err:
        print(f"[MongoDB] Save error: {m_err}")

def read_from_db(session_id=None):
    """Read prediction history from MongoDB Atlas."""
    items = []
    try:
        col = get_mongo_collection()
        if col is not None:
            query = {}
            if session_id:
                query['session_id'] = session_id
            cursor = col.find(query).sort('_id', -1).limit(100)
            for doc in cursor:
                items.append({
                    'text': doc.get('input_text') or doc.get('text', ''),
                    'prediction': doc.get('prediction', ''),
                    'confidence': float(doc.get('confidence', 0.0)),
                    'timestamp': doc.get('timestamp', ''),
                    'session_id': doc.get('session_id', '')
                })
            print(f"[MongoDB] Fetched {len(items)} records from MongoDB Atlas")
        else:
            print("[MongoDB] History fetch: MongoDB is not connected yet.")
    except Exception as m_err:
        print(f"[MongoDB] Read error: {m_err}")
    return items



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    img_dir = os.path.join(app.root_path, 'static', 'img')
    if os.path.exists(os.path.join(img_dir, 'logo.png')):
        return send_from_directory(img_dir, 'logo.png', mimetype='image/png')
    return ('', 204)

@app.route('/api')
def api_root():
    return jsonify({
        'name': 'TruthScan AI Fake News Detection API',
        'status': 'online',
        'endpoints': {
            'health': '/health',
            'predict': '/predict (POST)',
            'history': '/history (GET)',
            'about': '/about (GET)'
        }
    })

@app.route('/predict', methods=['GET', 'POST'])
@app.route('/api/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return jsonify({
            'status': 'online',
            'endpoint': '/predict',
            'method': 'POST',
            'message': 'Fake News Detection API is active.',
            'usage': 'Send a POST request with JSON: {"text": "article text here", "session_id": "optional"}'
        })
    try:
        data = request.get_json(silent=True) or {}
        text = data.get('text', '').strip()
        session_id = data.get('session_id', '').strip()
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        if len(text) < 10:
            return jsonify({'error': 'Text too short for accurate analysis'}), 400

        if detector and preprocessor:
            result = detector.predict(text)
        else:
            result = fallback_predict(text)

        ist_timestamp = get_ist_time()

        save_to_db(text, result['prediction'], result['confidence'], ist_timestamp, session_id)

        return jsonify(result)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to analyze text'}), 500

@app.route('/history', methods=['GET'])
@app.route('/api/history', methods=['GET'])
def history():

    try:
        session_id = request.args.get('session_id', '').strip()

        all_items = read_from_db(session_id=session_id)

        seen_texts = set()
        unique_items = []
        for item in all_items:
            text_key = item.get('text', '').strip().lower()
            if text_key and text_key not in seen_texts:
                seen_texts.add(text_key)
                unique_items.append(item)

        unique_items.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        final_list = unique_items[:50]

        print(f"Returning {len(final_list)} items for session_id={session_id!r}")
        resp = jsonify(final_list)
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma'] = 'no-cache'
        resp.headers['Expires'] = '0'
        return resp

    except Exception as e:
        print(f"History error: {e}")
        import traceback
        traceback.print_exc()
        resp = jsonify([])
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        return resp

@app.route('/health')
def health_check():
    mongo_active = False
    try:
        col = get_mongo_collection()
        mongo_active = col is not None
    except Exception:
        mongo_active = False

    return jsonify({
        'status': 'healthy',
        'database': 'mongodb' if mongo_active else 'disconnected',
        'mongodb_connected': mongo_active,
        'model_loaded': detector is not None,
        'preprocessor_loaded': preprocessor is not None
    })

@app.route('/about')
def about():
    return jsonify({
        'system': 'Fake News Detection System',
        'version': '1.0.0',
        'technologies': [
            'Machine Learning (Logistic Regression, Naive Bayes)',
            'Natural Language Processing',
            'TF-IDF Feature Extraction',
            'MongoDB Atlas (Cloud Database)',
            'Flask Web Framework',
            'HTML/CSS/JavaScript'
        ],
        'features': [
            'Text preprocessing and cleaning',
            'Stop word removal and lemmatization',
            'Feature extraction using TF-IDF',
            'Multiple classification algorithms',
            'Performance evaluation metrics',
            'Cloud-persistent history via MongoDB Atlas',
            'Web-based user interface'
        ]
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500


initialize_model()

if __name__ == '__main__':
    print("Starting Fake News Detection System...")

    if not os.path.exists('models'):
        os.makedirs('models')

    port = int(os.environ.get('PORT', 5000))
    print(f"Starting Flask server on port {port}...")
    app.run(host='0.0.0.0', port=port)
