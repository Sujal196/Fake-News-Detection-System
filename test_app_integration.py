#!/usr/bin/env python3
"""
Test App Integration with Large Dataset Model
"""

import os
import pickle
from data_preprocessing import TextPreprocessor

def test_app_model_loading():
    """Test how app.py loads and uses the model"""
    print('🔍 Testing App Model Integration...')
    print('=' * 50)
    
    try:
        # Test 1: Check if model files exist
        model_path = 'models/best_model_large.pkl'
        vectorizer_path = 'models/vectorizer_large.pkl'
        
        print(f'📁 Model file exists: {os.path.exists(model_path)}')
        print(f'📁 Vectorizer file exists: {os.path.exists(vectorizer_path)}')
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print('❌ Model files missing!')
            return False
        
        # Test 2: Load model directly (like app.py should do)
        print('\n📦 Loading model files...')
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        print(f'✅ Model loaded: {type(model).__name__}')
        print(f'✅ Vectorizer loaded: {type(vectorizer).__name__}')
        
        # Test 3: Test with FakeNewsDetector (app.py uses this)
        print('\n🔧 Testing FakeNewsDetector integration...')
        from ml_models import FakeNewsDetector
        
        detector = FakeNewsDetector()
        
        # This is what app.py tries to do
        try:
            detector.load_model(model_path, vectorizer_path)
            print('✅ FakeNewsDetector loaded model successfully')
        except Exception as e:
            print(f'❌ FakeNewsDetector failed: {e}')
            print('🔧 Need to fix FakeNewsDetector.load_model method')
        
        # Test 4: Direct prediction test
        print('\n🧪 Testing direct prediction...')
        test_text = "Scientists discover new breakthrough in cancer research"
        
        # Preprocess
        preprocessor = TextPreprocessor()
        processed_text = preprocessor.preprocess_text(test_text)
        
        # Vectorize
        text_vector = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        confidence = max(model.predict_proba(text_vector)[0])
        
        result = 'Real News' if prediction == 1 else 'Fake News'
        print(f'✅ Direct prediction: {result} (confidence: {confidence:.3f})')
        
        # Test 5: Test through FakeNewsDetector.predict method
        if hasattr(detector, 'predict'):
            try:
                result = detector.predict(test_text)
                print(f'✅ Detector prediction: {result["prediction"]} (confidence: {result["confidence"]:.3f})')
            except Exception as e:
                print(f'❌ Detector.predict failed: {e}')
        
        return True
        
    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()
        return False

def create_fixed_detector():
    """Create a fixed detector that works with the large dataset model"""
    print('\n🔧 Creating Fixed Detector...')
    print('=' * 50)
    
    # Create a new detector class that works with raw models
    class FixedFakeNewsDetector:
        def __init__(self):
            self.model = None
            self.vectorizer = None
            self.preprocessor = TextPreprocessor()
        
        def load_model(self, model_path, vectorizer_path):
            """Load raw model and vectorizer"""
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            print('✅ Fixed detector loaded model successfully')
        
        def predict(self, text):
            """Predict using the loaded model"""
            if self.model is None or self.vectorizer is None:
                raise ValueError("Model not loaded")
            
            # Preprocess text
            processed_text = self.preprocessor.preprocess_text(text)
            
            # Vectorize
            text_vector = self.vectorizer.transform([processed_text])
            
            # Predict
            prediction = self.model.predict(text_vector)[0]
            probabilities = self.model.predict_proba(text_vector)[0]
            
            result = {
                'prediction': 'Real News' if prediction == 1 else 'Fake News',
                'confidence': max(probabilities),
                'probabilities': {
                    'Fake News': probabilities[0],
                    'Real News': probabilities[1]
                }
            }
            
            return result
    
    # Test the fixed detector
    try:
        detector = FixedFakeNewsDetector()
        detector.load_model('models/best_model_large.pkl', 'models/vectorizer_large.pkl')
        
        # Test predictions
        test_cases = [
            "Scientists discover new breakthrough in cancer research",
            "SHOCKING miracle cure overnight secret revealed",
            "Government announces new economic policies",
            "ALIENS conspiracy theory found"
        ]
        
        print('🧪 Testing Fixed Detector:')
        for text in test_cases:
            result = detector.predict(text)
            print(f'  "{text[:30]}..." -> {result["prediction"]} (confidence: {result["confidence"]:.3f})')
        
        print('✅ Fixed detector working correctly!')
        return detector
        
    except Exception as e:
        print(f'❌ Fixed detector failed: {e}')
        return None

if __name__ == "__main__":
    # Test current integration
    current_works = test_app_model_loading()
    
    # Create and test fixed version
    fixed_detector = create_fixed_detector()
    
    if fixed_detector:
        print('\n🎉 SOLUTION: Use FixedFakeNewsDetector in app.py')
        print('📝 Update app.py to use the fixed detector class')
