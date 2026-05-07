#!/usr/bin/env python3
"""
Create Model That Actually Works Correctly
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from data_preprocessing import TextPreprocessor
import pickle

def create_working_model():
    print('🔧 Creating Model That Actually Works')
    print('=' * 50)
    
    # Load data
    fake_df = pd.read_csv('Fake.csv')
    true_df = pd.read_csv('True.csv')
    
    print(f'Fake.csv: {len(fake_df)} articles')
    print(f'True.csv: {len(true_df)} articles')
    
    # Create balanced small sample for testing
    fake_sample = fake_df.sample(1000, random_state=42)
    true_sample = true_df.sample(1000, random_state=42)
    
    # Clear labels
    fake_sample['label'] = 0  # Fake News
    true_sample['label'] = 1  # Real News
    
    # Combine
    combined_df = pd.concat([fake_sample, true_sample], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f'📊 Training on: {len(combined_df)} articles (balanced)')
    
    # Simple preprocessing
    preprocessor = TextPreprocessor()
    
    # Use only title for clearer patterns
    combined_df['text'] = combined_df['title'].fillna('')
    combined_df['processed_text'] = combined_df['text'].apply(
        lambda x: preprocessor.preprocess_text(str(x))
    )
    
    # Create features
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        stop_words='english',
        min_df=2,
        max_df=0.8
    )
    
    X = vectorizer.fit_transform(combined_df['processed_text'])
    y = combined_df['label']
    
    print(f'🔧 Features: {X.shape}')
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train with RandomForest (better for this task)
    print('🚀 Training RandomForest...')
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    )
    
    model.fit(X_train, y_train)
    
    # Test
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f'📊 Train accuracy: {train_score:.4f}')
    print(f'📊 Test accuracy: {test_score:.4f}')
    
    # Test with clear examples
    print('\n🧪 Testing with Clear Examples:')
    
    test_cases = [
        ("Scientists discover new cancer treatment", 1),  # Real
        ("SHOCKING miracle cure overnight", 0),  # Fake
        ("Government announces economic policy", 1),  # Real
        ("ALIENS found in backyard", 0),  # Fake
        ("Researchers publish study", 1),  # Real
        ("Celebrity reveals secret", 0),  # Fake
        ("Congress passes bill", 1),  # Real
        ("BREAKING shocking news", 0),  # Fake
    ]
    
    correct = 0
    for text, expected in test_cases:
        processed = preprocessor.preprocess_text(text)
        vector = vectorizer.transform([processed])
        pred = model.predict(vector)[0]
        confidence = max(model.predict_proba(vector)[0])
        
        result = 'Real News' if pred == 1 else 'Fake News'
        expected_label = 'Real News' if expected == 1 else 'Fake News'
        status = '✅' if pred == expected else '❌'
        
        if pred == expected:
            correct += 1
        
        print(f'  {status} "{text}" -> {result} (confidence: {confidence:.3f})')
        print(f'      Expected: {expected_label}')
    
    accuracy = (correct / len(test_cases)) * 100
    print(f'\n📊 Test Accuracy: {correct}/{len(test_cases)} ({accuracy:.1f}%)')
    
    if accuracy >= 75:
        print('🎉 Model is working correctly!')
        # Save model
        with open('models/correct_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        with open('models/correct_vectorizer.pkl', 'wb') as f:
            pickle.dump(vectorizer, f)
        
        print('✅ Saved: models/correct_model.pkl')
        print('✅ Saved: models/correct_vectorizer.pkl')
        
        # Update app.py
        update_app_py()
        
        return True
    else:
        print('❌ Model still not working properly')
        return False

def update_app_py():
    """Update app.py to use correct model"""
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    content = content.replace(
        "model_path = 'models/final_model.pkl'",
        "model_path = 'models/correct_model.pkl'"
    )
    content = content.replace(
        "vectorizer_path = 'models/final_vectorizer.pkl'",
        "vectorizer_path = 'models/correct_vectorizer.pkl'"
    )
    
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print('✅ Updated app.py to use correct_model.pkl')

if __name__ == "__main__":
    create_working_model()
