from ml_models import FakeNewsDetector
import sys

# Initialize detector and load the newly trained models
detector = FakeNewsDetector()
try:
    detector.load_model('models/final_model.pkl', 'models/final_vectorizer.pkl')
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Sample texts to test
real_news_sample = "The Federal Reserve raised interest rates by half a percentage point on Wednesday in an effort to combat inflation, marking the largest single increase since 2000. Chairman Jerome Powell stated that the economy remains robust despite global challenges."

fake_news_sample = "SHOCKING TRUTH REVEALED: Miracle cure for aging discovered overnight! Doctors hate this secret alien technology that makes you look 20 years younger instantly. Share this before the government deletes it!"

print("\n" + "="*50)
print("TESTING REAL NEWS")
print("="*50)
print(f"INPUT TEXT: {real_news_sample}\n")

real_result = detector.predict(real_news_sample)
print(f"PREDICTION: {real_result['prediction']}")
print(f"CONFIDENCE: {real_result['confidence'] * 100:.2f}%\n")
print("EXPLANATION:")
print(real_result['explanation'])


print("\n" + "="*50)
print("TESTING FAKE NEWS")
print("="*50)
print(f"INPUT TEXT: {fake_news_sample}\n")

fake_result = detector.predict(fake_news_sample)
print(f"PREDICTION: {fake_result['prediction']}")
print(f"CONFIDENCE: {fake_result['confidence'] * 100:.2f}%\n")
print("EXPLANATION:")
print(fake_result['explanation'])
