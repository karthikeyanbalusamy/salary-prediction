import pandas as pd
import joblib
import boto3
import numpy as np
import mlflow
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

S3_BUCKET = "kb-ml-bucket"
S3_KEY = "latest/model.pkl"

mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

df = pd.read_csv("data/processed/clean.csv")
X = df[['YearsExperience']]
y = df['Salary']

# Train Test Split]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Metrics
mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

# MLFlow Logging
with mlflow.start_run():
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse_train", mse_train)
    mlflow.log_metric("mse_test", mse_test)
    mlflow.log_metric("rmse_train", rmse_train)
    mlflow.log_metric("rmse_test", rmse_test)
    mlflow.log_metric("r2_train", r2_train)
    mlflow.log_metric("r2_test", r2_test)

    mlflow.sklearn.log_model(model, "model")

joblib.dump(model, "models/model.pkl")

# Upload to S3 for serving
s3 = boto3.client("s3")
s3.upload_file("models/model.pkl", S3_BUCKET, S3_KEY)

print("Model trained and uploaded to S3")
print(f"mse (test) : {mse_test}")
print(f"rmse (test): {rmse_test:.2f}")
print(f"r2 (test) : {r2_test}")
