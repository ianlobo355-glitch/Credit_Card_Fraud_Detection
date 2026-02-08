"""
app.py
Flask API for credit card fraud detection predictions with authentication.

Routes:
    GET  /login - Login page
    POST /login - Process login
    GET  /register - Registration page
    POST /register - Process registration
    GET  /dashboard - User dashboard (protected)
    GET  /admin - Admin dashboard (protected)
    GET  /logout - Logout user
    GET  / - Homepage (redirects to dashboard if logged in)
    POST /predict - Make fraud prediction (requires authentication)
    GET  /health - Check API status
    GET  /info - Get API information

Features:
    - User authentication with login/logout
    - Session management
    - Protected routes for authenticated users
    - Admin and regular user roles
    - Fraud detection predictions for logged-in users
    - Requires pre-trained model (run main.py first)

Prerequisites:
    1. Train model: python main.py
    2. This creates fraud_detection_model.pkl and scaler.pkl
"""

import logging
import json
import os
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_cors import CORS
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from model import load_model, predict_transaction
from adaptive_learning import (add_feedback, get_feedback_stats, retrain_model, 
                            apply_retrained_model, clear_feedback, 
                            create_model_backup, restore_model_from_backup)

# Initialize Flask app with templates folder
template_dir = os.path.abspath('templates1')
static_dir = os.path.abspath('templates1')
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir, static_url_path='/static')
CORS(app)

# Configure session
app.secret_key = 'credit_card_fraud_detection_secret_key_2026'
app.config['SESSION_TYPE'] = 'filesystem'

# User database file
USERS_DB_FILE = 'users.json'

# Helper functions for user management
def load_users():
    """Load users from JSON file."""
    if os.path.exists(USERS_DB_FILE):
        try:
            with open(USERS_DB_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_users(users):
    """Save users to JSON file."""
    with open(USERS_DB_FILE, 'w') as f:
        json.dump(users, f, indent=4)

def user_exists(username):
    """Check if user exists."""
    users = load_users()
    return username.lower() in users

def create_user(username, password, email='', role='user'):
    """Create a new user."""
    users = load_users()
    if username.lower() in users:
        return False, 'Username already exists'
    
    users[username.lower()] = {
        'username': username,
        'password': password,  # In production, use proper hashing!
        'email': email,
        'role': role,
        'created_at': datetime.now().isoformat(),
        'predictions': []
    }
    save_users(users)
    return True, 'User created successfully'

def verify_user(username, password):
    """Verify username and password."""
    users = load_users()
    user = users.get(username.lower())
    if user and user['password'] == password:
        return True, user
    return False, None

def get_user(username):
    """Get user by username."""
    users = load_users()
    return users.get(username.lower())

def save_prediction(username, prediction_data):
    """Save prediction to user's history."""
    users = load_users()
    user = users.get(username.lower())
    if user:
        user['predictions'].append({
            'features': prediction_data['features'],
            'prediction': prediction_data['prediction'],
            'prediction_label': prediction_data['prediction_label'],
            'confidence': prediction_data['confidence'],
            'timestamp': datetime.now().isoformat()
        })
        save_users(users)

# Authentication decorator
def login_required(f):
    """Decorator to require login."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    """Decorator to require admin role."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        user = get_user(session['username'])
        if user and user.get('role') == 'admin':
            return f(*args, **kwargs)
        return redirect(url_for('dashboard'))
    return decorated_function

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load the pre-trained model and scaler
try:
    model, scaler = load_model('fraud_detection_model.pkl', 'scaler.pkl')
    logger.info("✓ Model and scaler loaded successfully!")
except FileNotFoundError as e:
    logger.error(f"✗ Model files not found: {e}")
    logger.error("Please run 'python main.py' to train the model first!")
    model = None
    scaler = None

# Number of features expected (V1-V28 + Amount = 29 features)
EXPECTED_FEATURES = 29
FEATURE_NAMES = [f'V{i}' for i in range(1, 29)] + ['Amount']

# -----------------------------------------------------------------
# Feature mapping layer
# -----------------------------------------------------------------
# The trained ML model expects a fixed-length 29-element numeric vector
# (V1..V28, Amount). To keep the UI simple we accept a small set of
# user-friendly fields (amount, transaction_type, card_present, time)
# and map them to the model input. PCA features (V1..V28) are centered
# around 0 in the original dataset, so using 0 as a default is realistic.
# We apply lightweight heuristics to slightly adjust a few PCA-features
# based on the user inputs to provide a reasonable signal without
# exposing model internals. This abstraction keeps the model unchanged
# while allowing a simple UI for end users.
def map_user_input_to_features(amount: float,
                               transaction_type: str = 'Domestic',
                               card_present: bool = True,
                               txn_time: float | None = None) -> list:
    """Map simplified user inputs to the 29-length feature vector.

    Parameters
    - amount: transaction amount (float)
    - transaction_type: 'Domestic' or 'International'
    - card_present: True if card is present
    - txn_time: Unix timestamp or seconds-since-midnight (optional)

    Returns
    - features: list of 29 floats (V1..V28, Amount)

    Notes:
    - V1..V28 are set to 0 by default (PCA components are zero-centered).
    - Small, interpretable offsets are added to a few V* features to
      represent higher risk for international or card-not-present cases.
    - This mapping is intentionally simple and conservative so the
      original trained model remains unchanged and results are stable.
    """
    # Start with zero for PCA features (realistic default for standardized PCA components)
    v = [0.0] * 28

    # Heuristic adjustments (stronger fraud signal for risk factors)
    # International transactions are much riskier — strong signal
    if str(transaction_type).lower().startswith('int'):
        v[4] += 3.5   # V5 - strong international signal
        v[11] += 2.8  # V12
        v[3] += 2.0   # V4 - additional component

    # Card-not-present is a major fraud risk indicator
    if not card_present:
        v[19] += 3.2  # V20 - strong card-not-present signal
        v[22] += 2.5  # V23
        v[14] += 1.8  # V15 - additional component
    
    # Combined risk: international AND card-not-present is very suspicious
    if str(transaction_type).lower().startswith('int') and not card_present:
        v[8] += 2.0   # V9 - extra boost for combined risk

    # Large amounts raise suspicion
    try:
        amt = float(amount)
    except Exception:
        amt = 0.0
    if amt > 2000:
        v[0] += 2.5   # V1 
    elif amt > 1000:
        v[0] += 1.8   # V1 - moderate amount boost
    elif amt > 500:
        v[0] += 0.5   # V1 - small boost for medium amounts

    # Time-of-day can slightly influence suspicion (e.g., late-night)
    if txn_time is not None:
        # Accept either seconds-since-midnight or unix timestamp
        try:
            t = float(txn_time)
            # normalize: seconds in day
            if t > 86400:  # unix timestamp -> convert to seconds-since-midnight
                from datetime import datetime
                t = datetime.fromtimestamp(t).hour * 3600 + datetime.fromtimestamp(t).minute * 60
            hour = int((t % 86400) // 3600)
            if hour >= 0 and hour <= 5:
                # late-night transactions slightly riskier
                v[7] += 0.3  # V8
        except Exception:
            pass

    # Final feature vector: V1..V28 followed by Amount
    features = v + [amt]
    return features

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        
        if not username or not password:
            return render_template('LOGIN-MAIN.html', error='Username and password required'), 400
        
        verified, user = verify_user(username, password)
        if verified:
            session['username'] = username
            session['role'] = user.get('role', 'user')
            logger.info(f"✓ User '{username}' logged in successfully")
            
            # Redirect to admin or user dashboard based on role
            if user.get('role') == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('dashboard'))
        
        logger.warning(f"✗ Failed login attempt for username '{username}'")
        return render_template('LOGIN-MAIN.html', error='Invalid username or password'), 401
    
    return render_template('LOGIN-MAIN.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        # Validation
        if not username or not password:
            return render_template('REGISTER.html', error='Username and password required'), 400
        
        if len(username) < 3:
            return render_template('REGISTER.html', error='Username must be at least 3 characters'), 400
        
        if len(password) < 6:
            return render_template('REGISTER.html', error='Password must be at least 6 characters'), 400
        
        if password != confirm_password:
            return render_template('REGISTER.html', error='Passwords do not match'), 400
        
        success, message = create_user(username, password, email, role='user')
        if success:
            logger.info(f"✓ New user '{username}' registered successfully")
            return render_template('REGISTER.html', success='Registration successful! Please log in.'), 200
        
        logger.warning(f"✗ Registration failed for username '{username}': {message}")
        return render_template('REGISTER.html', error=message), 400
    
    return render_template('REGISTER.html')

@app.route('/logout')
def logout():
    """Handle user logout."""
    username = session.get('username', 'Unknown')
    session.clear()
    logger.info(f"✓ User '{username}' logged out")
    return redirect(url_for('login'))

# ==================== PROTECTED ROUTES ====================

@app.route('/')
def home():
    """Serve homepage - redirect to dashboard if logged in."""
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    """User dashboard with fraud detection interface."""
    username = session.get('username')
    user = get_user(username)
    return render_template('USER.html', username=username, prediction_count=len(user.get('predictions', [])))

@app.route('/admin')
@admin_required
def admin_dashboard():
    """Admin dashboard."""
    username = session.get('username')
    users = load_users()
    total_users = len(users)
    total_predictions = sum(len(u.get('predictions', [])) for u in users.values())
    
    return render_template('ADMIN.html', 
                         username=username, 
                         total_users=total_users,
                         total_predictions=total_predictions,
                         recent_users=list(users.values())[-10:])

# ==================== API ROUTES ==================== 


@app.route('/health', methods=['GET'])
def health_check():
    """Check API health status."""
    if model is None or scaler is None:
        return jsonify({'status': 'error', 'message': 'Model not loaded'}), 500
    return jsonify({'status': 'ok', 'message': 'API is running'}), 200

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """
    Make fraud prediction for a transaction.

    Backwards-compatible: accepts either the full model-level `features` list
    (existing clients) or the simplified user-friendly inputs:
      - amount (number)
      - transaction_type ("Domestic" or "International")
      - card_present (true/false)
      - time (optional timestamp or seconds-since-midnight)

    The simplified inputs are converted to the 29-element vector using
    `map_user_input_to_features` so the trained model remains unchanged.
    """
    if model is None or scaler is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

    try:
        data = request.get_json()
        if data is None:
            return jsonify({'error': 'Invalid JSON format'}), 400

        # 1) If caller provides raw model features, use them (backwards compatible)
        if 'features' in data:
            features = data['features']
            if not isinstance(features, list):
                return jsonify({'error': '"features" must be a list'}), 400
            if len(features) != EXPECTED_FEATURES:
                return jsonify({'error': f'Expected {EXPECTED_FEATURES} features, got {len(features)}'}), 400
        else:
            # 2) Accept simplified inputs and map them to model features
            amount = data.get('amount')
            transaction_type = data.get('transaction_type', 'Domestic')
            card_present = data.get('card_present', True)
            txn_time = data.get('time', None)

            # Normalize boolean-like card_present
            if isinstance(card_present, str):
                card_present = card_present.lower() in ('1', 'true', 'yes', 'y')

            # Map to the 29-feature vector
            features = map_user_input_to_features(amount, transaction_type, card_present, txn_time)

        # Convert to numpy array for prediction
        try:
            features_array = np.array(features, dtype=float)
        except Exception:
            return jsonify({'error': 'All feature values must be numeric'}), 400

        logger.debug(f"User '{session.get('username')}' submitted simplified features.")

        # Make prediction
        prediction, probabilities = predict_transaction(model, scaler, features_array)

        # Apply fraud risk rules: boost fraud score for high-risk transaction types
        fraud_prob = float(probabilities[1])
        is_international = str(data.get('transaction_type', '')).lower().startswith('int')
        is_card_not_present = str(data.get('card_present', 'true')).lower() in ('0', 'false', 'no')
        amount = float(data.get('amount', 0))
        
        # Rule 1: International + Card-Not-Present is very high risk
        if is_international and is_card_not_present:
            fraud_prob = max(fraud_prob, 0.75)  # Boost to at least 75% fraud probability
        # Rule 2: Card-Not-Present with high amount is risky
        elif is_card_not_present and amount > 1500:
            fraud_prob = max(fraud_prob, 0.60)  # Boost to at least 60%
        # Rule 3: International with high amount
        elif is_international and amount > 2000:
            fraud_prob = max(fraud_prob, 0.65)  # Boost to at least 65%

        # Recalculate prediction label and confidence based on adjusted fraud probability
        prediction = 1 if fraud_prob >= 0.5 else 0
        label = 'Potential Fraud' if prediction == 1 else 'Likely Normal'
        
        result = {
            'prediction': int(prediction),
            'prediction_label': label,
            'confidence': {
                'normal': 1.0 - fraud_prob,
                'fraud': fraud_prob
            },
            'fraud_probability': fraud_prob,
            'explanation': (
                "International + Card-Not-Present: high fraud risk" if (is_international and is_card_not_present)
                else "Card-Not-Present + High amount: fraud risk" if (is_card_not_present and amount > 1500)
                else "International + High amount: fraud risk" if (is_international and amount > 2000)
                else "Standard transaction risk assessment"
            ),
            # Model-level features (29 floats) returned so the frontend can submit them with feedback
            'model_features': list(map(float, features_array.tolist()))
        }

        # Save prediction to user history (preserve model-level 'features' schema)
        save_prediction(session.get('username'), {
            'features': list(map(float, features)),
            'prediction': result['prediction'],
            'prediction_label': result['prediction_label'],
            'confidence': result['confidence']
        })

        logger.info(f"✓ Prediction: {result['prediction_label']} (fraud_prob={result['fraud_probability']:.4f}) by user '{session.get('username')}'")
        return jsonify(result), 200

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/info', methods=['GET'])
def info():
    """Return API information."""
    return jsonify({
        'name': 'Credit Card Fraud Detection API',
        'version': '2.0',
        'description': 'Predicts whether a transaction is fraudulent (requires authentication)',
        'expected_features': EXPECTED_FEATURES,
        'feature_names': FEATURE_NAMES,
        'model_type': 'Random Forest Classifier',
        'authentication': 'Required for predictions',
        'routes': {
            'authentication': {
                '/login': 'GET/POST - User login',
                '/register': 'GET/POST - User registration',
                '/logout': 'GET - User logout'
            },
            'protected': {
                '/dashboard': 'GET - User dashboard',
                '/admin': 'GET - Admin dashboard (admin role required)',
                '/predict': 'POST - Make fraud prediction (requires auth)',
                '/user-stats': 'GET - Get user statistics'
            },
            'public': {
                '/health': 'GET - Check API status',
                '/info': 'GET - Get API information'
            }
        }
    }), 200

@app.route('/user-stats')
@login_required
def user_stats():
    """Get user statistics."""
    username = session.get('username')
    user = get_user(username)
    predictions = user.get('predictions', [])
    
    fraud_count = sum(1 for p in predictions if p['prediction'] == 1)
    normal_count = len(predictions) - fraud_count
    
    return jsonify({
        'username': username,
        'total_predictions': len(predictions),
        'fraud_detected': fraud_count,
        'normal_transactions': normal_count,
        'recent_predictions': predictions[-5:] if predictions else []
    }), 200

# ==================== ADAPTIVE LEARNING ROUTES ====================

@app.route('/feedback', methods=['POST'])
@login_required
def submit_feedback():
    """
    Submit feedback on a prediction (was it correct?).
    Used for adaptive learning.
    
    Expected payload:
    {
        'features': [array of 29 floats],
        'prediction': 0 or 1,
        'actual_label': 0 or 1 (true label)
    }
    """
    try:
        data = request.get_json()
        
        features = data.get('features')
        prediction = data.get('prediction')
        actual_label = data.get('actual_label')
        
        if features is None or prediction is None or actual_label is None:
            return jsonify({'error': 'Missing required fields: features, prediction, actual_label'}), 400
        
        if len(features) != 29:
            return jsonify({'error': 'Features must be 29 elements'}), 400
        
        # Record the feedback
        feedback_entry = add_feedback(features, prediction, actual_label, username=session.get('username', 'anonymous'))
        
        stats = get_feedback_stats()
        
        return jsonify({
            'success': True,
            'message': 'Feedback recorded for adaptive learning',
            'was_correct': feedback_entry['was_correct'],
            'feedback_stats': stats
        }), 200
        
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        return jsonify({'error': f'Feedback failed: {str(e)}'}), 500

@app.route('/feedback-stats', methods=['GET'])
def get_feedback_status():
    """Get statistics on collected feedback and adaptive learning progress."""
    stats = get_feedback_stats()
    return jsonify({
        'adaptive_learning': {
            'enabled': True,
            'total_feedback_entries': stats['total_feedback'],
            'correct_predictions': stats['correct_predictions'],
            'incorrect_predictions': stats['incorrect_predictions'],
            'feedback_accuracy': f"{stats['accuracy']:.2f}%",
            'ready_for_retraining': stats['total_feedback'] >= 10
        }
    }), 200

@app.route('/retrain', methods=['POST'])
@admin_required
def trigger_retrain():
    """
    Trigger model retraining using collected feedback (admin only).  
    SAFETY: Requires original training data + feedback. NO feedback-only mode.
    """
    global model, scaler
    
    stats = get_feedback_stats()
    
    if stats['total_feedback'] < 10:
        return jsonify({
            'error': f"Insufficient feedback for retraining. Need at least 10 examples, have {stats['total_feedback']}",
            'current_feedback': stats['total_feedback']
        }), 400
    
    try:
        logger.info(f"[ADMIN] User '{session.get('username')}' triggered model retraining")
        logger.info(f"Current feedback: {stats['total_feedback']} examples")
        
        # REQUIRED: Load original training data for safe retraining
        import pickle
        X_train_original = None
        y_train_original = None
        scaler_original = scaler
        
        if os.path.exists('training_data.pkl'):
            try:
                with open('training_data.pkl', 'rb') as f:
                    training_data = pickle.load(f)
                X_train_original = training_data.get('X_train')
                y_train_original = training_data.get('y_train')
                scaler_original = training_data.get('scaler', scaler)
                logger.info("✓ Loaded original training data for safe retraining")
            except Exception as e:
                logger.error(f"ERROR: Could not load training data: {e}")
                return jsonify({
                    'error': 'Training data file corrupted. Cannot safely retrain.',
                    'details': str(e)
                }), 500
        else:
            return jsonify({
                'error': 'Original training data not found (training_data.pkl missing)',
                'message': 'Cannot perform retraining without original data (feedback-only mode is disabled for safety)',
                'recovery': 'Re-run main.py to rebuild training data'
            }), 400
        
        # Perform retraining with original data (REQUIRED)
        new_model, new_scaler = retrain_model(X_train_original, y_train_original, scaler_original)
        
        if new_model is None:
            return jsonify({
                'error': 'Retraining failed - could not train model',
                'message': 'Check admin console logs for details'
            }), 500
        
        # Apply the new model with safety checks
        success = apply_retrained_model(new_model, new_scaler)
        
        if success:
            # Update global references
            model = new_model
            scaler = new_scaler
            
            # Clear feedback after successful retraining
            clear_feedback()
            
            return jsonify({
                'success': True,
                'message': 'Model successfully retrained (original data + feedback learning)',
                'feedback_used': stats['total_feedback'],
                'correct_predictions_learned': stats['correct_predictions'],
                'incorrect_predictions_corrected': stats['incorrect_predictions'],
                'training_mode': 'combined (original + feedback - SAFE)',
                'backup_location': 'fraud_detection_model.pkl.backup'
            }), 200
        else:
            return jsonify({
                'error': 'Failed to apply retrained model - model restored from backup',
                'details': 'Check admin console logs'
            }), 500
    except Exception as e:
        logger.error(f"ERROR in retrain: {e}")
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500
            
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        return jsonify({'error': f'Retraining failed: {str(e)}'}), 500

if __name__ == '__main__':
    if model is None:
        logger.error("\n" + "=" * 80)
        logger.error("ERROR: Model not loaded!")
        logger.error("=" * 80)
        logger.error("Please train the model first by running:")
        logger.error("  > python main.py")
        logger.error("=" * 80)
    else:
        # Create demo admin user if it doesn't exist
        if not user_exists('admin'):
            create_user('admin', 'admin123', 'admin@fraud-detection.local', role='admin')
            logger.info("✓ Demo admin account created: username='admin', password='admin123'")
        
        logger.info("\n" + "=" * 80)
        logger.info("Starting Credit Card Fraud Detection API with Authentication")
        logger.info("=" * 80)
        logger.info("\nAuthentication Routes:")
        logger.info("  GET  /login         - User login page")
        logger.info("  POST /login         - Process login")
        logger.info("  GET  /register      - User registration page")
        logger.info("  POST /register      - Process registration")
        logger.info("  GET  /logout        - User logout")
        logger.info("\nProtected Routes:")
        logger.info("  GET  /               - Home (redirects to dashboard)")
        logger.info("  GET  /dashboard     - User dashboard (requires auth)")
        logger.info("  GET  /admin         - Admin dashboard (admin role only)")
        logger.info("  POST /predict       - Make fraud prediction (requires auth)")
        logger.info("  GET  /user-stats    - Get user statistics (requires auth)")
        logger.info("\nAdaptive Learning Routes (admin):")
        logger.info("  POST /feedback      - Submit feedback on predictions")
        logger.info("  GET  /feedback-stats - View adaptive learning progress")
        logger.info("  POST /retrain       - Retrain model with collected feedback (admin)")
        logger.info("\nPublic Routes:")
        logger.info("  GET  /health       - Check API status")
        logger.info("  GET  /info         - Get API information")
        logger.info("\nDemo Credentials:")
        logger.info("  Admin  - username: admin, password: admin123")
        logger.info("\n" + "=" * 80)
        logger.info("Adaptive Learning enabled - Collect feedback to improve predictions!")
        logger.info("=" * 80 + "\n")
        app.run(debug=True, host='localhost', port=5000)
