#!/usr/bin/env python3
"""
Test Real Model Performance
"""

import pickle
import pandas as pd
from data_preprocessing import TextPreprocessor

def test_real_performance():
    print('🔍 Testing Actual Model Performance')
    print('=' * 50)
    
    # Load the actual model being used
    with open('models/final_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/final_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    print(f'✅ Model loaded: {type(model).__name__}')
    
    # Test with real data from both files
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    
    preprocessor = TextPreprocessor()
    
    # Test 10 samples from each
    print('\n🧪 Testing with Real Data:')
    
    fake_correct = 0
    real_correct = 0
    
    print('FAKE NEWS TESTS (should predict 0):')
    for i in range(10):
        text = f'{fake_df.iloc[i]["title"]} {fake_df.iloc[i]["text"]}'
        processed = preprocessor.preprocess_text(text)
        vector = vectorizer.transform([processed])
        pred = model.predict(vector)[0]
        
        if pred == 0:
            fake_correct += 1
        print(f'  Test {i+1}: {pred} (✅)' if pred == 0 else f'  Test {i+1}: {pred} (❌)')
    
    print(f'Fake News Accuracy: {fake_correct}/10 ({fake_correct*10}%)')
    
    print('\nREAL NEWS TESTS (should predict 1):')
    for i in range(10):
        text = f'{true_df.iloc[i]["title"]} {true_df.iloc[i]["text"]}'
        processed = preprocessor.preprocess_text(text)
        vector = vectorizer.transform([processed])
        pred = model.predict(vector)[0]
        
        if pred == 1:
            real_correct += 1
        print(f'  Test {i+1}: {pred} (✅)' if pred == 1 else f'  Test {i+1}: {pred} (❌)')
    
    print(f'Real News Accuracy: {real_correct}/10 ({real_correct*10}%)')
    
    total_accuracy = (fake_correct + real_correct) / 20 * 100
    print(f'\n📊 OVERALL ACCURACY: {total_accuracy:.1f}%')
    
    if total_accuracy < 80:
        print('❌ MODEL IS NOT WORKING PROPERLY!')
        print('Need to retrain with correct approach')
        return False
    else:
        print('✅ Model is working correctly')
        return True

if __name__ == "__main__":
    test_real_performance()
