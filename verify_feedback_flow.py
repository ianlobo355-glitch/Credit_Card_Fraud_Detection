"""
verify_feedback_flow.py

Uses Flask test_client to run an end-to-end feedback flow without starting the server:
1. Login as `testuser`
2. Make a prediction via `/predict`
3. Submit feedback to `/feedback`
4. Print `/feedback-stats` response

Run: python verify_feedback_flow.py
"""

from app import app
import json

print('Starting end-to-end feedback flow test...')

with app.test_client() as client:
    # 1) Login (follow redirects to ensure session cookie set)
    login_data = {'username': 'testuser', 'password': 'password123'}
    r = client.post('/login', data=login_data, follow_redirects=True)
    print('Login status:', r.status_code)

    # 2) Make a prediction (simple sample)
    payload = {
        'amount': 25.0,
        'transaction_type': 'Domestic',
        'card_present': True,
        'time': None
    }
    pr = client.post('/predict', json=payload)
    print('Predict status:', pr.status_code)
    try:
        prj = pr.get_json()
    except Exception as e:
        print('Failed to parse predict response:', e)
        prj = None

    if prj is None:
        print('Prediction failed, aborting test')
    else:
        print('Prediction result:', json.dumps(prj, indent=2))
        features = prj.get('model_features')
        prediction = prj.get('prediction')
        # 3) Submit feedback: mark as correct (actual_label == prediction)
        fb_payload = {
            'features': features,
            'prediction': prediction,
            'actual_label': prediction
        }
        fb = client.post('/feedback', json=fb_payload)
        print('Feedback submit status:', fb.status_code)
        try:
            fbj = fb.get_json()
            print('Feedback response:', json.dumps(fbj, indent=2))
        except:
            print('No JSON from feedback endpoint')

        # 4) Get feedback stats
        fs = client.get('/feedback-stats')
        print('Feedback-stats status:', fs.status_code)
        try:
            fsj = fs.get_json()
            print('Feedback-stats:', json.dumps(fsj, indent=2))
        except:
            print('No JSON from feedback-stats')

print('End-to-end feedback flow test complete.')
