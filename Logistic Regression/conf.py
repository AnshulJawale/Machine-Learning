import os, joblib

BASE_DIR = os.path.dirname(__file__)
CSV_DIR = os.path.join(BASE_DIR, "CSVs")
MODELS_DIR = os.path.join(BASE_DIR, "Models")

def join_csv(filename):
    return os.path.join(CSV_DIR, f"{filename}.csv")

def join_models(modelname):
    return os.path.join(MODELS_DIR, f"{modelname}")

def save_model(model, filename):
    model_path = os.path.join(MODELS_DIR, f"{filename}")
    joblib.dump(model, model_path)
