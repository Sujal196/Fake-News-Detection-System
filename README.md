# TruthScan — AI Fake News Detection System

TruthScan is an AI-powered web application that detects and classifies fake news in real time using Natural Language Processing (NLP) and Machine Learning. Trained on over 44,000 news articles, it achieves a 99.4% accuracy rate to help users verify information quickly and reliably.

---

### 🌐 Live Links
- **Live Demo (Frontend):** [https://fake-news-detection-system-omega.vercel.app/](https://fake-news-detection-system-omega.vercel.app/)
- **Backend API:** [https://fake-news-detection-system-8yn3.onrender.com](https://fake-news-detection-system-8yn3.onrender.com)

---

## 📌 Features

- **⚡ Instant Real-Time Analysis:** Classifies news articles as *Real* or *Fake* within seconds.
- **🎯 High Accuracy (99.4%):** Powered by an optimized Logistic Regression classifier and TF-IDF feature extraction.
- **📊 Confidence & Probability Breakdown:** Displays visual probability bars and confidence percentages for full transparency.
- **🧠 AI Explanation:** Highlights key linguistic cues, credible sources, and authenticity indicators.
- **🕒 Analysis History:** Automatically logs past predictions to MongoDB Atlas for easy tracking and review.
- **🎨 Modern Responsive UI:** Clean dark/light theme interface with glassmorphic styling and mobile support.

---

## 🛠️ Tech Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, CSS3 (Vanilla Glassmorphism), JavaScript (ES6+) |
| **Backend** | Python, Flask, Gunicorn (WSGI) |
| **Machine Learning & NLP** | scikit-learn, NLTK, TF-IDF Vectorizer |
| **Database** | MongoDB Atlas (Cloud) |
| **Hosting & Deployment** | Vercel (Frontend), Render (Backend API) |

---

## 🔍 How It Works

```
[ User Inputs Article ] 
          ↓
[ NLP Preprocessing: Cleaning, Tokenization, Lemmatization ]
          ↓
[ TF-IDF Vectorization (Feature Extraction) ]
          ↓
[ Machine Learning Model (Logistic Regression) ]
          ↓
[ Verdict + Confidence Score + AI Explanation ]
          ↓
[ Saved to MongoDB Atlas History ]
```

1. **Text Preprocessing:** Cleans input text by removing URLs, special characters, and stop words, then applies lemmatization.
2. **Feature Extraction:** Converts cleaned text into numerical features using TF-IDF (Term Frequency - Inverse Document Frequency).
3. **Classification & Storage:** The model predicts whether the article is Real or Fake, calculates confidence, and stores the record in MongoDB Atlas.

---

## 📊 Model Performance

Trained on the **ISOT Fake News Dataset** (44,898 total articles: 23,481 Fake, 21,417 Real).

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | **99.4%** | **99.4%** | **99.4%** | **99.4%** |
| Naive Bayes | 96.1% | 96.1% | 96.1% | 96.1% |

---

## 📁 Project Structure

```
Fake-News-Detection-System/
├── static/
│   ├── css/style.css          # Styling & theme variables
│   ├── js/script.js           # Client logic & API communication
│   └── img/                   # Logos and visual assets
├── templates/
│   └── index.html             # Main dashboard UI
├── models/
│   ├── final_model.pkl        # Trained classification model
│   └── preprocessor.pkl       # Fitted TF-IDF vectorizer
├── app.py                     # Flask application & API routes
├── data_preprocessing.py      # Text preprocessing pipeline
├── ml_models.py               # ML training and evaluation logic
├── requirements.txt           # Python dependencies
├── vercel.json                # Vercel deployment config
├── render.yaml                # Render service config
├── Procfile                   # Gunicorn process file
└── README.md                  # Project documentation
```
