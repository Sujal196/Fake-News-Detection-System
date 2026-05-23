from flask import Flask, render_template, request, jsonify
import os
import pickle
import sys
import sqlite3
import urllib.request
import urllib.error
import json
from datetime import datetime, timedelta, timezone
from data_preprocessing import TextPreprocessor
from ml_models import FakeNewsDetector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

detector = None
preprocessor = None

KVDB_URL = os.environ.get('KVDB_URL', "https://kvdb.io/YG9gdzbbU4PfdDAWHFmRNd/history")
KVDB_TOKEN = os.environ.get('KVDB_TOKEN', '').strip()
# Optional: name of header to use for the token (default: Authorization Bearer)
KVDB_AUTH_HEADER = os.environ.get('KVDB_AUTH_HEADER', '').strip()

def get_ist_time():
    ist_tz = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist_tz).strftime('%Y-%m-%d %H:%M:%S')

def get_kvdb_history():
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        if KVDB_TOKEN:
            if KVDB_AUTH_HEADER:
                headers[KVDB_AUTH_HEADER] = KVDB_TOKEN
            else:
                headers['Authorization'] = f'Bearer {KVDB_TOKEN}'

        req = urllib.request.Request(KVDB_URL, headers=headers)
        with urllib.request.urlopen(req, timeout=8) as response:
            data = response.read().decode('utf-8')
            if not data or data.strip() == '':
                return []
            return json.loads(data)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        print(f"KVDB GET error {e.code}: {e}")
        return None
    except Exception as e:
        print(f"Failed to read from kvdb: {e}")
        return None

def save_to_kvdb(item):
    try:
        # Read existing history (may return None on failure)
        current_history = get_kvdb_history() or []

        # Deduplicate and prepend
        current_history = [x for x in current_history if x.get('text', '').strip() != item['text'].strip()]
        current_history.insert(0, item)
        current_history = current_history[:100]

        data_bytes = json.dumps(current_history).encode('utf-8')

        # prepare headers with optional auth
        headers = {'Content-Type': 'application/json', 'User-Agent': 'Mozilla/5.0'}
        if KVDB_TOKEN:
            if KVDB_AUTH_HEADER:
                headers[KVDB_AUTH_HEADER] = KVDB_TOKEN
            else:
                headers['Authorization'] = f'Bearer {KVDB_TOKEN}'

        # simple retry loop for transient network issues
        attempts = 0
        max_attempts = 2
        while attempts < max_attempts:
            try:
                put_req = urllib.request.Request(
                    KVDB_URL,
                    data=data_bytes,
                    headers=headers,
                    method='PUT'
                )
                with urllib.request.urlopen(put_req, timeout=8) as response:
                    status = response.getcode()
                    print(f"KVDB PUT status: {status}")
                    print(f"Saved to KVDB: {item['text'][:50]}...")
                    return True
            except urllib.error.HTTPError as e:
                # non-transient (auth/permission) errors should not be retried
                print(f"KVDB PUT HTTPError {e.code}: {e}")
                return False
            except Exception as e:
                print(f"KVDB PUT attempt {attempts+1} failed: {e}")
                attempts += 1
                if attempts >= max_attempts:
                    print("KVDB PUT: all attempts failed")
                    return False
    except Exception as e:
        print(f"KVDB save error: {e}")
        return False

def get_db_path():
    if os.environ.get('VERCEL') == '1':
        return None
    base_dir = os.path.dirname(os.path.abspath(__file__))
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

        new_item = {
            'text': text,
            'prediction': result['prediction'],
            'confidence': float(result['confidence']),
            'timestamp': ist_timestamp,
            'session_id': session_id
        }

        is_vercel = os.environ.get('VERCEL') == '1'

        if is_vercel:
            saved = save_to_kvdb(new_item)
            print(f"Vercel KVDB save result: {saved}")
            # If KVDB save failed, attempt to persist locally as a best-effort
            # fallback so history is still available for the session.
            if not saved:
                # Use project-local DB file even on Vercel as a best-effort fallback.
                base_dir = os.path.dirname(os.path.abspath(__file__))
                db_path = get_db_path() or os.path.join(base_dir, 'predictions.db')
                if db_path:
                    try:
                        conn = sqlite3.connect(db_path, timeout=10.0)
                        conn.isolation_level = None
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT INTO history (input_text, prediction, confidence, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
                            (text, result['prediction'], float(result['confidence']), ist_timestamp, session_id)
                        )
                        conn.commit()
                        cursor.close()
                        conn.close()
                        print(f"Saved to SQLite (fallback): {text[:50]}... at {ist_timestamp}")
                    except Exception as db_err:
                        print(f"SQLite fallback error: {db_err}")
                        import traceback
                        traceback.print_exc()
        else:
            db_path = get_db_path()
            try:
                conn = sqlite3.connect(db_path, timeout=10.0)
                conn.isolation_level = None
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO history (input_text, prediction, confidence, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
                    (text, result['prediction'], float(result['confidence']), ist_timestamp, session_id)
                )
                conn.commit()
                cursor.close()
                conn.close()
                print(f"Saved to SQLite: {text[:50]}... at {ist_timestamp}")
            except Exception as db_err:
                print(f"SQLite error: {db_err}")
                import traceback
                traceback.print_exc()
            save_to_kvdb(new_item)

        return jsonify(result)
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to analyze text'}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        is_vercel = os.environ.get('VERCEL') == '1'
        session_id = request.args.get('session_id', '').strip()
        base_dir = os.path.dirname(os.path.abspath(__file__))

        all_items = []

        if is_vercel:
            # Try KVDB first (used on Vercel). If KVDB is unavailable or returns
            # None (e.g., permission/403 errors), fall back to local SQLite if
            # present so history still works during misconfiguration.
            kvdb_data = get_kvdb_history()
            if kvdb_data:
                all_items = kvdb_data
            else:
                # KVDB read failed; attempt to read local DB as a best-effort fallback.
                db_path = get_db_path()
                # if get_db_path() returns None (e.g., Vercel), try project-local file
                if not db_path:
                    db_path = os.path.join(base_dir, 'predictions.db')
                if db_path and os.path.exists(db_path):
                    try:
                        conn = sqlite3.connect(db_path, timeout=10.0)
                        conn.isolation_level = None
                        cursor = conn.cursor()
                        cursor.execute("SELECT input_text, prediction, confidence, timestamp, session_id FROM history ORDER BY id DESC")
                        rows = cursor.fetchall()
                        cursor.close()
                        conn.close()
                        print(f"Fetched {len(rows)} records from SQLite (fallback)")
                        for r in rows:
                            all_items.append({
                                'text': r[0],
                                'prediction': r[1],
                                'confidence': float(r[2]),
                                'timestamp': r[3],
                                'session_id': r[4] or ''
                            })
                    except Exception as db_read_err:
                        print(f"Database read error (fallback): {db_read_err}")
                        import traceback
                        traceback.print_exc()
        else:
            db_path = get_db_path()
            try:
                conn = sqlite3.connect(db_path, timeout=10.0)
                conn.isolation_level = None
                cursor = conn.cursor()
                cursor.execute("SELECT input_text, prediction, confidence, timestamp, session_id FROM history ORDER BY id DESC")
                rows = cursor.fetchall()
                cursor.close()
                conn.close()
                print(f"Fetched {len(rows)} records from SQLite")
                for r in rows:
                    all_items.append({
                        'text': r[0],
                        'prediction': r[1],
                        'confidence': float(r[2]),
                        'timestamp': r[3],
                        'session_id': r[4] or ''
                    })
            except Exception as db_read_err:
                print(f"Database read error: {db_read_err}")
                import traceback
                traceback.print_exc()

            kvdb_data = get_kvdb_history()
            if kvdb_data:
                existing_texts = {x.get('text', '').strip().lower() for x in all_items}
                for kitem in kvdb_data:
                    if kitem.get('text', '').strip().lower() not in existing_texts:
                        all_items.append(kitem)

        if session_id and all_items:
            filtered = [item for item in all_items if item.get('session_id', '') == session_id]
            all_items = filtered

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
