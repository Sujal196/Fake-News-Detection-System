#!/usr/bin/env python3
"""
Diagnose Label Inversion Issue in Large Dataset Model
"""

import pandas as pd
import numpy as np
from data_preprocessing import TextPreprocessor

def diagnose_label_issue():
    """Diagnose why model predicts everything as Fake News"""
    print('🔍 Diagnosing Label Inversion Issue...')
    print('=' * 60)
    
    # Load the original datasets to check label assignment
    print('📁 Loading original datasets...')
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    
    print(f'Fake.csv: {len(fake_df)} articles')
    print(f'True.csv: {len(true_df)} articles')
    
    # Check original label assignment in training script
    print('\n🔍 Checking label assignment in training...')
    print('Fake.csv should be labeled as 0 (Fake News)')
    print('True.csv should be labeled as 1 (Real News)')
    
    # Load training results to see what happened
    import pickle
    with open('models/training_results.pkl', 'rb') as f:
        results = pickle.load(f)
    
    print('\n📊 Training Results:')
    for model_name, result in results.items():
        print(f'  {model_name}: F1={result["f1_score"]:.4f}')
    
    # Check model predictions on training data
    with open('models/best_model_large.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/vectorizer_large.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Test with small sample of original data
    print('\n🧪 Testing with original data patterns...')
    
    # Create test samples from original data
    test_samples = []
    
    # Add some fake news samples
    fake_samples = fake_df.head(3)
    for _, row in fake_samples.iterrows():
        test_samples.append({
            'text': f"{row['title']} {row['text']}",
            'expected': 0,  # Fake News
            'source': 'Fake.csv'
        })
    
    # Add some real news samples  
    real_samples = true_df.head(3)
    for _, row in real_samples.iterrows():
        test_samples.append({
            'text': f"{row['title']} {row['text']}",
            'expected': 1,  # Real News
            'source': 'True.csv'
        })
    
    # Preprocess and test
    preprocessor = TextPreprocessor()
    
    print('\n🧪 Test Results:')
    print('-' * 50)
    
    correct = 0
    total = len(test_samples)
    
    for i, sample in enumerate(test_samples):
        # Preprocess
        processed_text = preprocessor.preprocess_text(sample['text'])
        
        # Vectorize
        text_vector = vectorizer.transform([processed_text])
        
        # Predict
        prediction = model.predict(text_vector)[0]
        confidence = max(model.predict_proba(text_vector)[0])
        
        # Determine result
        result = 'Real News' if prediction == 1 else 'Fake News'
        expected = 'Real News' if sample['expected'] == 1 else 'Fake News'
        
        # Check if correct
        is_correct = prediction == sample['expected']
        if is_correct:
            correct += 1
        
        status = '✅' if is_correct else '❌'
        
        print(f'  {status} Sample {i+1}: {result} (confidence: {confidence:.3f})')
        print(f'      Expected: {expected}')
        print(f'      Source: {sample["source"]}')
        print(f'      Match: {"YES" if is_correct else "NO"}')
    
    accuracy = (correct / total) * 100
    print(f'\n📊 Overall: {correct}/{total} correct ({accuracy:.1f}% accuracy)')
    
    # Analysis
    print('\n🔍 Label Analysis:')
    if accuracy == 0:
        print('❌ CRITICAL: Model predicts opposite of expected labels!')
        print('   This indicates label inversion during training.')
        print('   Check if Fake.csv and True.csv labels were swapped.')
    elif accuracy < 50:
        print('⚠️  SERIOUS: Model performance is worse than random!')
        print('   Major issue with label assignment or model training.')
    else:
        print('✅ Model is working correctly for these samples.')
    
    # Check if the issue is systematic
    predictions = []
    for sample in test_samples:
        processed_text = preprocessor.preprocess_text(sample['text'])
        text_vector = vectorizer.transform([processed_text])
        pred = model.predict(text_vector)[0]
        predictions.append(pred)
    
    unique_preds = np.unique(predictions)
    print(f'\n🔍 Unique predictions: {unique_preds}')
    
    if len(unique_preds) == 1:
        print('❌ Model only predicts one class!')
        if unique_preds[0] == 0:
            print('   Model only predicts "Fake News"')
            print('   This suggests True.csv samples were labeled as 0 during training.')
        else:
            print('   Model only predicts "Real News"')
            print('   This suggests Fake.csv samples were labeled as 1 during training.')
    
    print('\n💡 Recommendations:')
    print('1. Check label assignment in train_large_dataset.py')
    print('2. Verify Fake.csv is labeled as 0, True.csv as 1')
    print('3. Re-run training with corrected labels')
    
    return accuracy

if __name__ == "__main__":
    diagnose_label_issue()
