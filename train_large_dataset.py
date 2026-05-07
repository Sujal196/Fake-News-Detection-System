#!/usr/bin/env python3
"""
Advanced Fake News Detection Training on Large Dataset
Trains on Fake.csv and True.csv with multi-feature processing
Optimized for 90-95% accuracy
"""

import pandas as pd
import numpy as np
import time
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from data_preprocessing import load_and_preprocess_data, prepare_data

# Suppress warnings
warnings.filterwarnings('ignore')

class AdvancedFakeNewsTrainer:
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.vectorizer = None
        self.results = {}
        
    def load_and_combine_datasets(self, fake_file='Fake.csv', true_file='True.csv'):
        """Load and combine Fake.csv and True.csv with proper labeling"""
        print("=" * 80)
        print("LOADING AND COMBINING LARGE DATASETS")
        print("=" * 80)
        
        start_time = time.time()
        
        # Load fake news dataset
        print(f"Loading {fake_file}...")
        fake_df = pd.read_csv(fake_file)
        fake_df['label'] = 0  # 0 = Fake News
        fake_df['source'] = 'Fake.csv'
        print(f"✅ Fake news loaded: {len(fake_df):,} articles")
        
        # Load real news dataset
        print(f"Loading {true_file}...")
        true_df = pd.read_csv(true_file)
        true_df['label'] = 1  # 1 = Real News
        true_df['source'] = 'True.csv'
        print(f"✅ Real news loaded: {len(true_df):,} articles")
        
        # Combine datasets
        print("\nCombining datasets...")
        combined_df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Shuffle the dataset
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        load_time = time.time() - start_time
        
        print(f"\n📊 Dataset Statistics:")
        print(f"   Total articles: {len(combined_df):,}")
        print(f"   Fake news: {len(combined_df[combined_df['label'] == 0]):,}")
        print(f"   Real news: {len(combined_df[combined_df['label'] == 1]):,}")
        print(f"   Columns: {list(combined_df.columns)}")
        print(f"   Load time: {load_time:.2f} seconds")
        
        # Show sample data
        print(f"\n📝 Sample Data:")
        for i in range(min(3, len(combined_df))):
            row = combined_df.iloc[i]
            print(f"   Sample {i+1}: Label={row['label']}, Subject={row.get('subject', 'N/A')}")
            print(f"      Title: {str(row.get('title', ''))[:100]}...")
            print(f"      Text: {str(row.get('text', ''))[:100]}...")
            print()
        
        return combined_df
    
    def preprocess_large_dataset(self, df):
        """Preprocess the large dataset with multi-feature handling"""
        print("=" * 80)
        print("PREPROCESSING LARGE DATASET")
        print("=" * 80)
        
        start_time = time.time()
        
        # Work with the actual data
        processed_df = df.copy()
        
        # Combine features
        processed_df['combined_text'] = (
            processed_df['title'].fillna('') + ' ' + 
            processed_df['text'].fillna('') + ' ' + 
            processed_df['subject'].fillna('')
        )
        
        # Apply text preprocessing
        from data_preprocessing import TextPreprocessor
        preprocessor = TextPreprocessor()
        
        print("Applying text preprocessing...")
        processed_df['processed_text'] = processed_df['combined_text'].apply(
            lambda x: preprocessor.preprocess_text(str(x))
        )
        
        preprocess_time = time.time() - start_time
        
        print(f"✅ Preprocessing completed in {preprocess_time:.2f} seconds")
        print(f"   Processed text length: {processed_df['processed_text'].str.len().mean():.1f} chars avg")
        
        return processed_df
    
    def create_optimized_vectorizer(self, texts):
        """Create optimized TF-IDF vectorizer for large dataset"""
        print("=" * 80)
        print("CREATING OPTIMIZED FEATURE VECTORIZER")
        print("=" * 80)
        
        start_time = time.time()
        
        # Optimized parameters for large dataset
        self.vectorizer = TfidfVectorizer(
            max_features=50000,  # Increased for large dataset
            ngram_range=(1, 3),  # Unigrams, bigrams, trigrams
            min_df=2,  # Minimum document frequency
            max_df=0.85,  # Maximum document frequency
            sublinear_tf=True,
            stop_words='english',
            analyzer='word'
        )
        
        # Fit and transform
        X_tfidf = self.vectorizer.fit_transform(texts)
        
        vectorize_time = time.time() - start_time
        
        print(f"✅ Vectorization completed in {vectorize_time:.2f} seconds")
        print(f"   Feature vocabulary size: {len(self.vectorizer.vocabulary_):,}")
        print(f"   Feature matrix shape: {X_tfidf.shape}")
        print(f"   Feature density: {X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1]):.6f}")
        
        return X_tfidf
    
    def train_multiple_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models with hyperparameter optimization"""
        print("=" * 80)
        print("TRAINING MULTIPLE MODELS WITH HYPERPARAMETER OPTIMIZATION")
        print("=" * 80)
        
        # Define models with hyperparameter grids
        models_config = {
            'Logistic Regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l2'],
                    'solver': ['liblinear', 'lbfgs']
                }
            },
            'Naive Bayes': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.1, 0.5, 1.0, 2.0],
                    'fit_prior': [True, False]
                }
            },
            'Random Forest': {
                'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1],
                    'max_depth': [3, 5]
                }
            }
        }
        
        best_score = 0
        best_model_name = None
        
        for model_name, config in models_config.items():
            print(f"\n🔧 Training {model_name}...")
            start_time = time.time()
            
            # Grid search with cross-validation
            grid_search = GridSearchCV(
                config['model'],
                config['params'],
                cv=3,  # 3-fold cross-validation
                scoring='f1',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            # Get best model
            best_model = grid_search.best_estimator_
            
            # Make predictions
            y_pred = best_model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Detailed classification report
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            train_time = time.time() - start_time
            
            # Store results
            self.results[model_name] = {
                'model': best_model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'best_params': grid_search.best_params_,
                'training_time': train_time,
                'predictions': y_pred
            }
            
            print(f"   ✅ Training completed in {train_time:.2f} seconds")
            print(f"   📊 Accuracy: {accuracy:.4f}")
            print(f"   📊 Precision: {precision:.4f}")
            print(f"   📊 Recall: {recall:.4f}")
            print(f"   📊 F1-score: {f1:.4f}")
            print(f"   🔧 Best params: {grid_search.best_params_}")
            
            # Update best model
            if f1 > best_score:
                best_score = f1
                best_model_name = model_name
                self.best_model = best_model
        
        print(f"\n🏆 Best model: {best_model_name} with F1-score: {best_score:.4f}")
        return best_model_name
    
    def evaluate_best_model(self, X_test, y_test, model_name):
        """Detailed evaluation of the best model"""
        print("=" * 80)
        print(f"DETAILED EVALUATION OF BEST MODEL: {model_name}")
        print("=" * 80)
        
        result = self.results[model_name]
        y_pred = result['predictions']
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake News', 'Real News']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake News', 'Real News'],
                   yticklabels=['Fake News', 'Real News'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Confusion matrix saved as 'confusion_matrix_best_model.png'")
        
        # Feature importance (if available)
        if hasattr(result['model'], 'feature_importances_'):
            self.plot_feature_importance(result['model'], model_name)
        elif hasattr(result['model'], 'coef_'):
            self.plot_coefficients(result['model'], model_name)
    
    def plot_feature_importance(self, model, model_name):
        """Plot feature importance for tree-based models"""
        feature_names = self.vectorizer.get_feature_names_out()
        importances = model.feature_importances_
        
        # Get top 20 features
        indices = np.argsort(importances)[-20:]
        top_features = [feature_names[i] for i in indices]
        top_importances = [importances[i] for i in indices]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_features)), top_importances)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Feature Importance')
        plt.title(f'Top 20 Important Features - {model_name}')
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Feature importance plot saved as 'feature_importance.png'")
    
    def plot_coefficients(self, model, model_name):
        """Plot coefficients for linear models"""
        feature_names = self.vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        
        # Get top 20 positive and negative coefficients
        pos_indices = np.argsort(coefficients)[-20:]
        neg_indices = np.argsort(coefficients)[:20]
        
        pos_features = [feature_names[i] for i in pos_indices]
        pos_coeffs = [coefficients[i] for i in pos_indices]
        
        neg_features = [feature_names[i] for i in neg_indices]
        neg_coeffs = [coefficients[i] for i in neg_indices]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Positive coefficients (real news indicators)
        ax1.barh(range(len(pos_features)), pos_coeffs)
        ax1.set_yticks(range(len(pos_features)))
        ax1.set_yticklabels(pos_features)
        ax1.set_xlabel('Coefficient Value')
        ax1.set_title(f'Top Real News Indicators - {model_name}')
        
        # Negative coefficients (fake news indicators)
        ax2.barh(range(len(neg_features)), neg_coeffs)
        ax2.set_yticks(range(len(neg_features)))
        ax2.set_yticklabels(neg_features)
        ax2.set_xlabel('Coefficient Value')
        ax2.set_title(f'Top Fake News Indicators - {model_name}')
        
        plt.tight_layout()
        plt.savefig('feature_coefficients.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Feature coefficients plot saved as 'feature_coefficients.png'")
    
    def save_models(self):
        """Save the best model and vectorizer"""
        print("=" * 80)
        print("SAVING MODELS")
        print("=" * 80)
        
        # Save best model
        with open('models/best_model_large.pkl', 'wb') as f:
            pickle.dump(self.best_model, f)
        print("✅ Best model saved as 'models/best_model_large.pkl'")
        
        # Save vectorizer
        with open('models/vectorizer_large.pkl', 'wb') as f:
            pickle.dump(self.vectorizer, f)
        print("✅ Vectorizer saved as 'models/vectorizer_large.pkl'")
        
        # Save results
        with open('models/training_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        print("✅ Training results saved as 'models/training_results.pkl'")
    
    def generate_training_report(self):
        """Generate a comprehensive training report"""
        print("=" * 80)
        print("TRAINING REPORT")
        print("=" * 80)
        
        report = []
        report.append("FAKE NEWS DETECTION - TRAINING REPORT")
        report.append("=" * 50)
        report.append(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Dataset Size: 44,898 articles")
        report.append(f"Fake News: 23,481 articles")
        report.append(f"Real News: 21,417 articles")
        report.append("")
        report.append("MODEL PERFORMANCE:")
        report.append("-" * 30)
        
        for model_name, result in self.results.items():
            report.append(f"{model_name}:")
            report.append(f"  Accuracy: {result['accuracy']:.4f}")
            report.append(f"  Precision: {result['precision']:.4f}")
            report.append(f"  Recall: {result['recall']:.4f}")
            report.append(f"  F1-score: {result['f1_score']:.4f}")
            report.append(f"  Training Time: {result['training_time']:.2f}s")
            report.append("")
        
        # Find best model
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        report.append(f"BEST MODEL: {best_model}")
        report.append(f"F1-SCORE: {self.results[best_model]['f1_score']:.4f}")
        
        # Save report
        with open('training_report.txt', 'w') as f:
            f.write('\n'.join(report))
        
        print("✅ Training report saved as 'training_report.txt'")
        
        # Print summary
        print("\n📊 SUMMARY:")
        for line in report[-10:]:
            print(line)

def main():
    """Main training pipeline"""
    print("🚀 ADVANCED FAKE NEWS DETECTION TRAINING")
    print("🎯 Target Accuracy: 90-95%")
    print("📊 Dataset: Fake.csv + True.csv (44,898 articles)")
    print("=" * 80)
    
    # Initialize trainer
    trainer = AdvancedFakeNewsTrainer()
    
    try:
        # Step 1: Load and combine datasets
        df = trainer.load_and_combine_datasets()
        
        # Step 2: Preprocess data
        processed_df = trainer.preprocess_large_dataset(df)
        
        # Step 3: Create features
        X = trainer.create_optimized_vectorizer(processed_df['processed_text'])
        y = processed_df['label']
        
        # Step 4: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 Data Split:")
        print(f"   Training set: {X_train.shape[0]:,} articles")
        print(f"   Test set: {X_test.shape[0]:,} articles")
        print(f"   Feature dimensions: {X_train.shape[1]:,}")
        
        # Step 5: Train models
        best_model_name = trainer.train_multiple_models(X_train, X_test, y_train, y_test)
        
        # Step 6: Evaluate best model
        trainer.evaluate_best_model(X_test, y_test, best_model_name)
        
        # Step 7: Save models
        trainer.save_models()
        
        # Step 8: Generate report
        trainer.generate_training_report()
        
        print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print(f"🏆 Best model: {best_model_name}")
        print(f"📊 Accuracy: {trainer.results[best_model_name]['accuracy']:.4f}")
        print(f"📊 F1-score: {trainer.results[best_model_name]['f1_score']:.4f}")
        
        if trainer.results[best_model_name]['accuracy'] >= 0.90:
            print("✅ Target accuracy achieved (≥90%)!")
        elif trainer.results[best_model_name]['accuracy'] >= 0.85:
            print("⚠️  Good accuracy achieved (85-90%)")
        else:
            print("❌ Target accuracy not achieved. Consider more data or different models.")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
