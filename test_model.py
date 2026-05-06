#!/usr/bin/env python3
"""
Test Large Dataset Model
"""

import pickle
import os
from data_preprocessing import TextPreprocessor

def test_large_model():
    """Test the large dataset model with known examples"""
    print('🔍 Testing Large Dataset Model...')
    print('=' * 50)
    
    # Check if model files exist
    if not os.path.exists('models/best_model_large.pkl') or not os.path.exists('models/vectorizer_large.pkl'):
        print('❌ Large dataset model files not found!')
        print('📊 Please run: python train_large_dataset.py')
        return
    
    try:
        # Load model and vectorizer
        with open('models/best_model_large.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('models/vectorizer_large.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        print('✅ Model and vectorizer loaded successfully')
        print(f'📈 Model type: {type(model).__name__}')
        
        # Test with known examples
        test_cases = [
            ('Scientists discover new cancer treatment breakthrough', 1),  # Should be Real
            ('SHOCKING miracle cure overnight secret', 0),  # Should be Fake
            ('Government announces new economic policies', 1),  # Should be Real
            ('ALIENS conspiracy theory found', 0),  # Should be Fake
            ('Researchers publish study in Nature journal', 1),  # Should be Real
            ('BREAKING celebrity reveals shocking cure', 0),  # Should be Fake
            ('Federal Reserve maintains interest rates', 1),  # Should be Real
            ('Miracle investment cryptocurrency overnight', 0),  # Should be Fake
            ('University announces research funding', 1),  # Should be Real
            ('Conspiracy government coverup revealed', 0)  # Should be Fake
        ]
        
        print('\n🧪 Testing predictions:')
        print('-' * 50)
        
        correct = 0
        total = len(test_cases)
        
        for i, (text, expected) in enumerate(test_cases, 1):
            # Preprocess text
            preprocessor = TextPreprocessor()
            processed_text = preprocessor.preprocess_text(text)
            
            # Vectorize
            text_vector = vectorizer.transform([processed_text])
            
            # Predict
            prediction = model.predict(text_vector)[0]
            confidence = max(model.predict_proba(text_vector)[0])
            
            result = 'Real News' if prediction == 1 else 'Fake News'
            status = '✅' if prediction == expected else '❌'
            
            if prediction == expected:
                correct += 1
                
            print(f'  {status} Test {i}: "{text[:40]}..." -> {result} (confidence: {confidence:.3f})')
        
        accuracy = (correct / total) * 100
        print('\n' + '-' * 50)
        print(f'📊 Results: {correct}/{total} correct ({accuracy:.1f}% accuracy)')
        
        if accuracy >= 90:
            print('🎉 Model is performing EXCELLENT!')
        elif accuracy >= 70:
            print('✅ Model is performing GOOD!')
        else:
            print('⚠️  Model needs improvement!')
            
        # Test with app.py integration
        print('\n🔗 Testing app.py integration...')
        try:
            # Simulate what app.py does
            from ml_models import FakeNewsDetector
            app_detector = FakeNewsDetector()
            app_detector.load_model('models/best_model_large.pkl', 'models/vectorizer_large.pkl')
            
            test_text = "Scientists discover new breakthrough in cancer research"
            result = app_detector.predict(test_text)
            print(f'📱 App integration: "{result["prediction"]}" (confidence: {result["confidence"]:.3f})')
            
        except Exception as e:
            print(f'❌ App integration error: {e}')
            
    except Exception as e:
        print(f'❌ Error testing model: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_large_model()
