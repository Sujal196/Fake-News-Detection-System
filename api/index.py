import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from data_preprocessing import TextPreprocessor
from ml_models import FakeNewsDetector

app = Flask(__name__)

detector = None
preprocessor = None

def initialize_model():
    global detector, preprocessor
    
    try:
        model_path = 'models/best_model.pkl'
        vectorizer_path = 'models/vectorizer.pkl'
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            print("Loading pre-trained model...")
            detector = FakeNewsDetector()
            detector.load_model(model_path, vectorizer_path)
        else:
            print("No pre-trained model found. Using fallback only.")
            detector = None
        
        preprocessor = TextPreprocessor()
        print("Model initialized successfully!")
        
    except Exception as e:
        print(f"Error initializing model: {e}")
        detector = None
        preprocessor = TextPreprocessor()

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
    return jsonify({
        'message': 'Fake News Detection API',
        'endpoints': {
            'predict': 'POST /predict - Analyze text for fake news',
            'health': 'GET /health - Check API status'
        }
    })

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
        
        return jsonify(result)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Failed to analyze text'}), 500

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': detector is not None,
        'preprocessor_loaded': preprocessor is not None
    })

# Initialize model on import
initialize_model()

# Vercel serverless handler
def handler(environ, start_response):
    return app(environ, start_response)
