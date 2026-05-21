import pickle

try:
    with open('models/training_results.pkl', 'rb') as f:
        results = pickle.load(f)
    print("Real Model Accuracies:")
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy:  {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall:    {result['recall']:.4f}")
        print(f"  F1 Score:  {result['f1_score']:.4f}")
except Exception as e:
    print(f"Error loading results: {e}")
