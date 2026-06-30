import joblib

models = [
    "logistic.pkl",
    "knn.pkl",
    "svc.pkl",
    "rf.pkl",
    "gb.pkl",
    "ann.pkl"
]

for model in models:
    try:
        joblib.load(f"models/{model}")
        print(f"✅ {model}")
    except Exception as e:
        print(f"❌ {model}")
        print(e)
        print("-"*50)