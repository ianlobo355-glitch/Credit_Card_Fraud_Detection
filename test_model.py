import joblib
import numpy as np

# Test normal transaction
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

# Test 1: Normal transaction (all zeros)
normal = np.zeros(29).reshape(1, -1)
normal_scaled = scaler.transform(normal)
pred = model.predict(normal_scaled)[0]
result1 = "FRAUD" if pred else "NORMAL"
print(f'Test 1 - Normal transaction (zeros): {result1} (expected NORMAL)')

# Test 2: Fraud example (International + Card-Not-Present)
fraud = np.zeros(29)
fraud[0] = 2.0
fraud[1] = 3.0
fraud_scaled = scaler.transform(fraud.reshape(1, -1))
pred = model.predict(fraud_scaled)[0]
result2 = "FRAUD" if pred else "NORMAL"
print(f'Test 2 - Fraud example (Intl+CNP): {result2} (expected FRAUD or NORMAL)')
