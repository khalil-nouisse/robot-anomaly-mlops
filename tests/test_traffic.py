# test_traffic.py
import requests
import random
import time

url = "http://localhost:8000/predict"

print("Sending 50 fake sequences to the API to trigger the anomaly counter...")
for i in range(50):
    # Your model expects sequences of exactly shape (250, 130)
    fake_sequence = [[random.uniform(-5.0, 5.0) for _ in range(130)] for _ in range(250)]
    
    response = requests.post(url, json={"sequence": fake_sequence})
    result = response.json()
    
    print(f"Request {i+1} status: {result['status']} | Anomaly Score: {result['anomaly_score']}")
    
    time.sleep(0.5) # Wait half a second between requests
