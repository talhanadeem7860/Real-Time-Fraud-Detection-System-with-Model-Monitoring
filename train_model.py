# train_model.py
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import joblib
import mlflow

print("[INFO] Generating synthetic transaction data...")
# Generate synthetic data
n_samples = 10000
data = {
    'amount': np.random.lognormal(3, 1, n_samples),
    'hour': np.random.randint(0, 24, n_samples),
    'merchant_category': np.random.randint(0, 10, n_samples),
    'user_age': np.random.randint(18, 70, n_samples),
}
df = pd.DataFrame(data)
# Create a simple fraud rule
df['is_fraud'] = ((df['amount'] > 150) & (df['hour'] < 6)).astype(int)

# Split data
X = df.drop('is_fraud', axis=1)
y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("[INFO] Training LightGBM model...")
# Train model
with mlflow.start_run():
    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"Model F1 Score: {f1:.4f}")
    
    # Log with MLflow
    mlflow.log_param("model_type", "LightGBM")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.lightgbm.log_model(model, "model")
    print("[INFO] Model and metrics logged to MLflow.")

# Save model and reference data
model_filename = 'fraud_model.pkl'
joblib.dump(model, model_filename)
print(f"[INFO] Model saved to {model_filename}")

reference_data_filename = 'reference_data.csv'
X_train.to_csv(reference_data_filename, index=False)
print(f"[INFO] Reference data saved to {reference_data_filename}")