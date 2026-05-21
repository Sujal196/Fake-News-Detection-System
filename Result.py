import pickle

with open('models/training_results.pkl', 'rb')as f:
    model = pickle.load(f)
print(model)