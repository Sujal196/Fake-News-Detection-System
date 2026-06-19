from flask import Flask, render_template, request, jsonify
import os
import sys
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from data_preprocessing import TextPreprocessor
from ml_models import FakeNewsDetector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

detector = None
preprocessor = None

def get_ist_time():
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist_tz).strftime('%Y-%m-%d %H:%M:%S')

def get_db_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if os.environ.get('VERCEL') == '1':
        return os.path.join(tempfile.gettempdir(), 'predictions.db')
    return os.path.join(base_dir, 'predictions.db')

def init_db():
    try:
        db_path = get_db_path()
        if not db_path:
            return
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT
            )
        ''')
        try:
            cursor.execute("ALTER TABLE history ADD COLUMN session_id TEXT")
        except sqlite3.OperationalError:
            pass
        conn.commit()
        conn.close()
        print(f"Database initialized at: {db_path}")
    except Exception as e:
        print(f"Database init error: {e}")

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
    """Save a prediction result to the local SQLite database."""
    try:
        db_path = get_db_path()
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.isolation_level = None
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO history (input_text, prediction, confidence, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            (text, prediction, float(confidence), timestamp, session_id)
        )
        conn.commit()
        cursor.close()
        conn.close()
        print(f"Saved to SQLite: {text[:50]}... at {timestamp}")
    except Exception as db_err:
        print(f"SQLite save error: {db_err}")
        import traceback
        traceback.print_exc()

def read_from_db():
    """Read all prediction history from the local SQLite database."""
    items = []
    try:
        db_path = get_db_path()
        if not db_path or not os.path.exists(db_path):
            return items
        conn = sqlite3.connect(db_path, timeout=10.0)
        conn.isolation_level = None
        cursor = conn.cursor()
        cursor.execute(
            "SELECT input_text, prediction, confidence, timestamp, session_id FROM history ORDER BY id DESC"
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        print(f"Fetched {len(rows)} records from SQLite")
        for r in rows:
            items.append({
                'text': r[0],
                'prediction': r[1],
                'confidence': float(r[2]),
                'timestamp': r[3],
                'session_id': r[4] or ''
            })
    except Exception as db_err:
        print(f"SQLite read error: {db_err}")
        import traceback
        traceback.print_exc()
    return items

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
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
def history():
    try:
        session_id = request.args.get('session_id', '').strip()

        all_items = read_from_db()

        if session_id and all_items:
            all_items = [item for item in all_items if item.get('session_id', '') == session_id]

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
    return jsonify({
        'status': 'healthy',
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
            'Flask Web Framework',
            'HTML/CSS/JavaScript'
        ],
        'features': [
            'Text preprocessing and cleaning',
            'Stop word removal and lemmatization',
            'Feature extraction using TF-IDF',
            'Multiple classification algorithms',
            'Performance evaluation metrics',
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

    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)
