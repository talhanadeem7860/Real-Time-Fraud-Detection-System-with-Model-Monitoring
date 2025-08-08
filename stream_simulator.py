# stream_simulator.py
import requests
import json
import time
import numpy as np
import pandas as pd

API_ENDPOINT = "http://127.0.0.1:5000/predict"
LOG_FILE = "live_data_log.csv"

def generate_transaction(is_drift=False):
    """Generates a single transaction, with an option to introduce drift."""
    amount = np.random.lognormal(3, 1)
    # Introduce drift by increasing the average transaction amount
    if is_drift:
        amount *= 2.5 
        
    return {
        'amount': amount,
        'hour': np.random.randint(0, 24),
        'merchant_category': np.random.randint(0, 10),
        'user_age': np.random.randint(18, 70),
    }

# Clear log file at start
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)

print("[INFO] Starting transaction stream simulator...")
print("[INFO] For the first 100 transactions, data will be normal.")
print("[INFO] After 100 transactions, data drift will be introduced.")

drift_active = False
for i in range(500):
    if i > 100 and not drift_active:
        print("\n[ALERT] Introducing data drift into the stream!\n")
        drift_active = True

    transaction = generate_transaction(is_drift=drift_active)
    
    # Log the transaction features to a CSV file
    pd.DataFrame([transaction]).to_csv(LOG_FILE, mode='a', header=not os.path.exists(LOG_FILE), index=False)
    
    try:
        # Send to API and get prediction
        response = requests.post(API_ENDPOINT, data=json.dumps(transaction))
        response.raise_for_status()
        prediction = response.json()
        print(f"Transaction {i+1}: Amount={transaction['amount']:.2f}, Fraud Prob={prediction['fraud_probability']:.4f}")
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Could not connect to API: {e}")
        print("[INFO] Is the prediction_api.py server running?")
        break
        
    time.sleep(1)