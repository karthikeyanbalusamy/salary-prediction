from fastapi import FastAPI
import joblib
import boto3
import numpy as np
import os

# ---------------------------
# CHANGE THESE IF NEEDED
# ---------------------------
S3_BUCKET = "kb-ml-bucket"
S3_KEY = "latest/model.pkl"
LOCAL_MODEL_PATH = "models/model.pkl"
# ---------------------------

app = FastAPI()

def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        s3 = boto3.client("s3")
        s3.download_file(S3_BUCKET, S3_KEY, LOCAL_MODEL_PATH)

# Download model ONCE at startup
download_model()

model = joblib.load(LOCAL_MODEL_PATH)

@app.post("/predict")
def predict(yr_exp: float):
    X = np.array([[yr_exp]])
    salary = model.predict(X)[0]
    return {"Years_Experience": yr_exp,
            "Predicted Salary":float(salary)}
