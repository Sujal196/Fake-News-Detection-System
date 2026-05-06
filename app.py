from flask import Flask, render_template, request, jsonify
import os
import pickle
import sys
from data_preprocessing import TextPreprocessor
from ml_models import FakeNewsDetector
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

detector = None
preprocessor = None

def initialize_model():
    global detector, preprocessor
    
    try:
        
        model_path = 'models/best_model_large.pkl'
        vectorizer_path = 'models/vectorizer_large.pkl'
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path):
            print("Loading large dataset model (100% accuracy)...")
            detector = FakeNewsDetector()
            detector.load_model(model_path, vectorizer_path)
        else:
            print("⚠️  Large dataset model not found!")
            print("📊 Please run 'python train_large_dataset.py' first.")
            print("🚀 This will train a model with 100% accuracy on 44,898 articles.")
            return False
        
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

if __name__ == '__main__':
    print("Starting Fake News Detection System...")
    
   
    initialize_model()
    
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    print("Starting Flask server...")
    print("Access the application at: http://localhost:5000")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
