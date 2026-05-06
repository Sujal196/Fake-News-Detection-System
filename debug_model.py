#!/usr/bin/env python3
"""
Debug Large Dataset Model Issues
"""

import pickle
import pandas as pd
import numpy as np
from data_preprocessing import TextPreprocessor

def debug_model():
    """Debug the large dataset model to find prediction issues"""
    print('🔍 Debugging Large Dataset Model...')
    print('=' * 50)
    
    # Load training results to see what happened
    with open('models/training_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    print('📊 Training Results:')
    for model_name, result in results.items():
        print(f'  {model_name}: F1={result["f1_score"]:.4f}, Accuracy={result["accuracy"]:.4f}')
    
    # Load model and vectorizer
    with open('models/best_model_large.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/vectorizer_large.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    print(f'✅ Loaded model: {type(model).__name__}')
    
    # Load actual data to test with
    print('\n📁 Loading test data...')
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    
    # Create balanced test set
    test_fake = fake_df.sample(n=5, random_state=42)
    test_real = true_df.sample(n=5, random_state=42)
    test_df = pd.concat([test_fake, test_real])
    test_df['label'] = [0]*5 + [1]*5  # 0=Fake, 1=Real
    
    print(f'📊 Test set: {len(test_df)} articles (5 fake, 5 real)')
    
    # Preprocess
    preprocessor = TextPreprocessor()
    test_df['combined_text'] = test_df['title'] + ' ' + test_df['text'] + ' ' + test_df['subject']
    test_df['processed_text'] = test_df['combined_text'].apply(preprocessor.preprocess_text)
    
    # Vectorize
    X_test = vectorizer.transform(test_df['processed_text'])
    y_test = test_df['label']
    
    # Predict
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    print('\n🧪 Prediction Results:')
    print('-' * 50)
    correct = 0
    total = len(test_df)
    
    for i, (idx, row) in enumerate(test_df.iterrows()):
        pred = predictions[i]
        actual = y_test.iloc[i]
        confidence = max(probabilities[i])
        result = 'Real News' if pred == 1 else 'Fake News'
        expected = 'Real News' if actual == 1 else 'Fake News'
        status = '✅' if pred == actual else '❌'
        
        if pred == actual:
            correct += 1
        
        print(f'  {status} {i+1}: {result} (confidence: {confidence:.3f})')
        print(f'      Expected: {expected}')
        print(f'      Title: {row["title"][:50]}...')
        print()
    
    accuracy = (correct / total) * 100
    print(f'\n📊 Final Results: {correct}/{total} correct ({accuracy:.1f}% accuracy)')
    
    if accuracy >= 90:
        print('🎉 Model is performing EXCELLENT!')
    elif accuracy >= 70:
        print('✅ Model is performing GOOD!')
    else:
        print('⚠️  Model needs investigation!')
    
    # Check for label distribution issues
    print('\n🔍 Label Distribution Analysis:')
    print(f'  Predictions - Fake: {sum(predictions == 0)}, Real: {sum(predictions == 1)}')
    print(f'  Actual - Fake: {sum(y_test == 0)}, Real: {sum(y_test == 1)}')
    
    # Check if model is always predicting same class
    unique_preds = np.unique(predictions)
    if len(unique_preds) == 1:
        print(f'⚠️  WARNING: Model only predicts class {unique_preds[0]}!')
        print('   This indicates training or data imbalance issues!')
    
    return accuracy

if __name__ == "__main__":
    debug_model()
