from flask import Flask, render_template, request, jsonify
import os
import pickle
import sys
import sqlite3
from data_preprocessing import TextPreprocessor
from ml_models import FakeNewsDetector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

detector = None
preprocessor = None

def get_db_path():
    if os.environ.get('VERCEL') == '1':
        return '/tmp/predictions.db'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'predictions.db')

def init_db():
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                input_text TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Failed to initialize database: {e}")

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
            print(f"⚠️  Model not found at {model_path}!")
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
        return {
            'prediction': 'Fake News',
            'confidence': min(0.9, 0.5 + fake_score * 0.1),
            'probabilities': {
                'Fake News': min(0.9, 0.5 + fake_score * 0.1),
                'Real News': max(0.1, 0.5 - fake_score * 0.1)
            }
        }
    else:
        return {
            'prediction': 'Real News',
            'confidence': min(0.9, 0.5 + real_score * 0.1),
            'probabilities': {
                'Fake News': max(0.1, 0.5 - real_score * 0.1),
                'Real News': min(0.9, 0.5 + real_score * 0.1)
            }
        }

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        if len(text) < 10:
            return jsonify({'error': 'Text too short for accurate analysis'}), 400
        
        if detector and preprocessor:
            result = detector.predict(text)
        else:
            result = fallback_predict(text)
        
        try:
            conn = sqlite3.connect(get_db_path())
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO history (input_text, prediction, confidence) VALUES (?, ?, ?)",
                (text, result['prediction'], float(result['confidence']))
            )
            conn.commit()
            conn.close()
        except Exception as db_err:
            print(f"Database error: {db_err}")
            
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Failed to analyze text'}), 500

@app.route('/history', methods=['GET'])
def history():
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        cursor.execute("SELECT id, input_text, prediction, confidence, timestamp FROM history ORDER BY id DESC LIMIT 50")
        rows = cursor.fetchall()
        conn.close()
        
        data = []
        for r in rows:
            data.append({
                'id': r[0],
                'text': r[1],
                'prediction': r[2],
                'confidence': r[3],
                'timestamp': r[4]
            })
        return jsonify(data)
    except Exception as e:
        print(f"History error: {e}")
        return jsonify({'error': 'Failed to retrieve history'}), 500

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
