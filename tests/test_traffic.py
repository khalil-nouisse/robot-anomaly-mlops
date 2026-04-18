# test_traffic.py
import requests
import random
import time

URL = "http://localhost:8000/predict"

def send_test_traffic(url: str = URL, num_requests: int = 50, sleep_s: float = 0.5):
    print(f"Sending {num_requests} fake sequences to the API to trigger the anomaly counter...")
    for i in range(num_requests):
        # Your model expects sequences of exactly shape (250, 130)
        fake_sequence = [[random.uniform(-5.0, 5.0) for _ in range(130)] for _ in range(250)]
        response = requests.post(url, json={"sequence": fake_sequence}, timeout=10)
        response.raise_for_status()
        result = response.json()

        print(f"Request {i+1} status: {result['status']} | Anomaly Score: {result['anomaly_score']}")
        time.sleep(sleep_s)  # Wait between requests


if __name__ == "__main__":
    send_test_traffic()
