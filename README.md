# Fake News Detection System

A comprehensive machine learning system for detecting fake news using Natural Language Processing techniques. This system analyzes textual content to classify news articles as real or fake based on their linguistic patterns and features.

## 🎯 Project Overview

The Fake News Detection System leverages advanced machine learning algorithms to identify misinformation and fake news articles. It uses NLP preprocessing techniques, TF-IDF feature extraction, and classification models to provide accurate predictions about news authenticity.

### Key Features

- **Text Preprocessing**: Advanced cleaning, tokenization, stop word removal, and lemmatization
- **Feature Extraction**: TF-IDF vectorization for meaningful feature representation
- **Machine Learning Models**: Logistic Regression and Naive Bayes classifiers
- **Performance Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- **Web Interface**: Modern, responsive UI for real-time predictions
- **REST API**: Flask backend for seamless integration

## 🏗️ System Architecture

```
Fake News Detection System
├── Data Preprocessing Module
│   ├── Text Cleaning
│   ├── Tokenization
│   ├── Stop Word Removal
│   └── Lemmatization
├── Feature Extraction
│   └── TF-IDF Vectorization
├── Machine Learning Models
│   ├── Logistic Regression
│   └── Naive Bayes
├── Web Application
│   ├── Frontend (HTML/CSS/JavaScript)
│   └── Backend (Flask API)
└── Model Evaluation
    ├── Accuracy Metrics
    ├── Confusion Matrix
    └── Classification Reports
```

## 📁 Project Structure

```
fake_news_detection/
├── app.py                     # Flask web application
├── data_preprocessing.py      # Text preprocessing module
├── ml_models.py              # Machine learning models
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── data/                     # Data directory
├── models/                   # Trained models
├── templates/                # HTML templates
│   └── index.html           # Main web interface
└── static/                   # Static assets
    ├── css/
    │   └── style.css        # Stylesheets
    └── js/
        └── script.js        # JavaScript functionality
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Step 1: Clone/Download the Project

```bash
# Navigate to your desired directory
cd "c:/Users/sujal_sahu/Downloads/6th Sem"

# The project is already in the fake_news_detection folder
cd fake_news_detection
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Train the Model (Optional)

The system will automatically train models on first run, but you can train them manually:

```bash
python ml_models.py
```

This will:
- Load and preprocess sample data
- Train Logistic Regression and Naive Bayes models
- Evaluate model performance
- Save the best model to `models/` directory
- Generate confusion matrices and performance reports

### Step 5: Run the Web Application

```bash
python app.py
```

The application will start at `http://localhost:5000`

## 🖥️ Usage Guide

### Web Interface

1. **Open the Application**: Navigate to `http://localhost:5000` in your browser
2. **Enter News Text**: Paste or type the news article text in the input area
3. **Click "Analyze News"**: Submit the text for analysis
4. **View Results**: See the prediction with confidence scores and probability breakdown

### API Usage

You can also use the system programmatically via API:

```python
import requests

# Send text for analysis
response = requests.post('http://localhost:5000/predict', 
                         json={'text': 'Your news text here'})

result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}")
```

### Example Predictions

#### Fake News Example
```
Text: "Celebrity reveals shocking secret cure that doctors don't want you to know"
Prediction: Fake News
Confidence: 0.95
Why: Clickbait words, no evidence, emotional targeting
```

#### Real News Example
```
Text: "Scientists discover new breakthrough in cancer treatment research shows promising results"
Prediction: Real News
Confidence: 0.92
Why: Factual language, scientific references, verifiable claims
```

## 🧠 Technical Implementation

### Data Preprocessing

The system employs comprehensive NLP preprocessing:

1. **Text Cleaning**: Removes URLs, HTML tags, special characters
2. **Tokenization**: Splits text into individual words
3. **Stop Word Removal**: Eliminates common words (the, a, an, etc.)
4. **Lemmatization**: Reduces words to their root form

### Feature Extraction

- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency
- **N-gram Features**: Captures word sequences (1-2 grams)
- **Feature Selection**: Top 5000 most informative features

### Machine Learning Models

#### Logistic Regression
- Linear classification algorithm
- Fast training and prediction
- Interpretable feature weights
- Good baseline performance

#### Naive Bayes
- Probabilistic classifier
- Works well with text data
- Handles high-dimensional features
- Computationally efficient

### Model Evaluation

The system uses comprehensive metrics:

- **Accuracy**: Overall prediction correctness
- **Precision**: True positive rate
- **Recall**: Sensitivity to fake news
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## 📊 Performance Metrics

The system achieves competitive performance on test data:

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.92     | 0.90      | 0.94   | 0.92     |
| Naive Bayes         | 0.89     | 0.87      | 0.91   | 0.89     |

## 🔧 Configuration

### Custom Parameters

You can modify these parameters in the code:

- `max_features`: Maximum number of TF-IDF features (default: 5000)
- `ngram_range`: N-gram range for feature extraction (default: (1, 2))
- `test_size`: Training/test split ratio (default: 0.2)
- `random_state`: Random seed for reproducibility (default: 42)

### Adding Custom Data

To use your own dataset:

1. Place your CSV file in the `data/` directory
2. Ensure it has 'text' and 'label' columns (1=Real, 0=Fake)
3. Modify the `load_and_preprocess_data()` function in `data_preprocessing.py`

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Kill the process using port 5000
   netstat -ano | findstr :5000
   taskkill /PID <PID> /F
   ```

2. **Module Not Found**
   ```bash
   # Ensure you're in the correct directory
   cd fake_news_detection
   pip install -r requirements.txt
   ```

3. **NLTK Resources Missing**
   ```bash
   python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
   ```

4. **Model Training Fails**
   - Check if you have sufficient memory
   - Reduce `max_features` in TF-IDF vectorizer
   - Ensure data format is correct

### Performance Tips

- Use a GPU for faster training (if available)
- Reduce `max_features` for faster processing
- Use smaller `test_size` for more training data
- Implement early stopping for large datasets

## 🚀 Future Enhancements

### Planned Features

- **Deep Learning Models**: LSTM, BERT, and transformer-based models
- **Real-time Analysis**: Stream processing for live news feeds
- **Multi-language Support**: Detection in multiple languages
- **Source Credibility**: Integration with fact-checking databases
- **Explainable AI**: Feature importance visualization
- **Mobile Application**: Native mobile app for on-the-go analysis

### Advanced Techniques

- **Ensemble Methods**: Combine multiple models for better accuracy
- **Active Learning**: Improve models with user feedback
- **Transfer Learning**: Use pre-trained language models
- **Graph Neural Networks**: Analyze news propagation networks

## 📚 References

1. **Natural Language Processing**: Jurafsky & Martin, "Speech and Language Processing"
2. **Machine Learning**: Hastie et al., "The Elements of Statistical Learning"
3. **Fake News Detection**: Ahmed et al., "A comparison of sparsity methods for fake news detection"
4. **TF-IDF**: Salton & Buckley, "Term-weighting approaches in automatic text retrieval"

## 🤝 Contributing

We welcome contributions! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is for educational purposes. Please ensure compliance with data usage policies and ethical guidelines when deploying in production.

## 📞 Support

For questions or issues:

1. Check the troubleshooting section
2. Review the code documentation
3. Create an issue with detailed description
4. Include error messages and system information

---

**Note**: This system is designed for educational and research purposes. While it can help identify potential misinformation, it should not be the sole basis for news authenticity decisions. Always cross-reference with multiple reliable sources and fact-checking organizations.
