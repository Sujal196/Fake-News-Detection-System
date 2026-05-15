import pandas as pd
import numpy as np
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Set NLTK data path to a writable directory on Vercel
nltk_data_path = os.path.join('/tmp', 'nltk_data')
if not os.path.exists(nltk_data_path):
    try:
        os.makedirs(nltk_data_path)
    except:
        pass

nltk.data.path.append(nltk_data_path)

def download_nltk_data():
    try:
        nltk.download('punkt', download_dir=nltk_data_path, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_path, quiet=True)
        nltk.download('wordnet', download_dir=nltk_data_path, quiet=True)
    except Exception as e:
        print(f"NLTK download warning: {e}")

download_nltk_data()

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
            
        text = text.lower()
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'<.*?>', '', text)
        
        text = re.sub(r'\S+@\S+', '', text)
        
        text = re.sub(r'\d{3}[-.]?\d{3}[-.]?\d{4}', '', text)
        
        text = re.sub(r'[^\w\s\.\!\?\,\:\;]', '', text)
        
        text = re.sub(r'\b(?!(?:19|20)\d{2}\b)\d+\b', '', text)
        
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_remove_stopwords(self, text):
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words]
        return tokens
    
    def lemmatize_tokens(self, tokens):
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def preprocess_text(self, text):
        cleaned_text = self.clean_text(text)
        
        tokens = self.tokenize_and_remove_stopwords(cleaned_text)
        
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        
        processed_text = ' '.join(lemmatized_tokens)
        
        return processed_text

def load_and_preprocess_data(file_path=None):
    if file_path and pd.io.common.file_exists(file_path):
        df = pd.read_csv(file_path)
        
        # Handle multi-feature dataset with title, text, subject, date
        if 'title' in df.columns and 'text' in df.columns:
            # Combine title and text for better feature extraction
            df['combined_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
            
            # Add subject as additional context if available
            if 'subject' in df.columns:
                df['combined_text'] = df['subject'].fillna('') + ' ' + df['combined_text']
            
            # Extract date features if available
            if 'date' in df.columns:
                df['date_features'] = extract_date_features(df['date'])
            else:
                df['date_features'] = ''
                
            # Final combined text
            df['final_text'] = df['combined_text'] + ' ' + df['date_features']
            
        elif 'text' in df.columns:
            # If only text column exists
            df['final_text'] = df['text'].fillna('')
        else:
            raise ValueError("Dataset must contain at least 'text' column")
            
    else:
        raise ValueError("No dataset file provided. Please provide a CSV file path or use the large dataset training script.")
        print("📊 For large dataset training, run: python train_large_dataset.py")
        print("🚀 This will use Fake.csv and True.csv files with 44,898 articles.")
    
    preprocessor = TextPreprocessor()
    
    df['processed_text'] = df['final_text'].apply(preprocessor.preprocess_text)
    
    return df

def extract_date_features(date_series):
    """Extract useful features from date column"""
    try:
        # Convert to datetime if not already
        dates = pd.to_datetime(date_series, errors='coerce')
        
        # Extract temporal features
        features = []
        for date in dates:
            if pd.notna(date):
                # Add temporal context as text
                temporal_info = f"year_{date.year} month_{date.month} day_{date.weekday()}"
                features.append(temporal_info)
            else:
                features.append('')
        return features
    except:
        return [''] * len(date_series)

def extract_features_tfidf(texts, max_features=50000):
    """Optimized TF-IDF for large datasets"""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 3),
        min_df=2,  # Increased for large dataset
        max_df=0.85,  # Adjusted for large dataset
        sublinear_tf=True,
        stop_words='english',
        analyzer='word'
    )
    features = vectorizer.fit_transform(texts)
    return features, vectorizer

def prepare_data(df):
    X = df['processed_text']
    y = df['label']
    
    X_tfidf, tfidf_vectorizer = extract_features_tfidf(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, tfidf_vectorizer

if __name__ == "__main__":
    print("Loading and preprocessing data...")
    df = load_and_preprocess_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Sample processed text: {df['processed_text'].iloc[0]}")
    
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(df)
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")