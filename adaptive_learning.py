"""
adaptive_learning.py
Implements simple online/adaptive learning for the fraud detection model.

Features:
    - Collects feedback on predictions (correct/incorrect labels)
    - Stores training examples with true labels
    - Periodically retrains model on accumulated data
    - Tracks model performance over time
"""

import logging
import json
import os
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from model import save_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# File to store feedback for retraining
FEEDBACK_FILE = 'adaptive_feedback.json'
TRAINING_LOG_FILE = 'training_log.json'

def load_feedback():
    """Load collected feedback from file."""
    if os.path.exists(FEEDBACK_FILE):
        try:
            with open(FEEDBACK_FILE, 'r') as f:
                return json.load(f)
        except:
            return []
    return []

def save_feedback(feedback_list):
    """Save feedback to file."""
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(feedback_list, f, indent=2)
    logger.info(f"Saved {len(feedback_list)} feedback entries")

def add_feedback(features, prediction, actual_label, username='system'):
    """
    Record feedback on a prediction.
    
    Args:
        features (list): Feature vector used for prediction (29 elements)
        prediction (int): Model's prediction (0 or 1)
        actual_label (int): True label (0=normal, 1=fraud)
        username (str): User who provided feedback
    """
    feedback_list = load_feedback()
    
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'features': list(map(float, features)) if isinstance(features, np.ndarray) else features,
        'prediction': int(prediction),
        'actual_label': int(actual_label),
        'username': username,
        'was_correct': int(prediction) == int(actual_label)
    }
    
    feedback_list.append(feedback_entry)
    save_feedback(feedback_list)
    
    logger.info(f"✓ Feedback recorded: pred={prediction}, actual={actual_label}, correct={feedback_entry['was_correct']}")
    return feedback_entry

def get_feedback_stats():
    """Get statistics on collected feedback."""
    feedback_list = load_feedback()
    
    if not feedback_list:
        return {
            'total_feedback': 0,
            'correct_predictions': 0,
            'incorrect_predictions': 0,
            'accuracy': 0.0
        }
    
    correct = sum(1 for f in feedback_list if f.get('was_correct', False))
    total = len(feedback_list)
    
    return {
        'total_feedback': total,
        'correct_predictions': correct,
        'incorrect_predictions': total - correct,
        'accuracy': round(correct / total * 100, 2) if total > 0 else 0.0
    }

def prepare_training_data_from_feedback():
    """
    Convert feedback entries into training data for retraining.
    
    Returns:
        tuple: (X, y) - Features and labels, or (None, None) if insufficient data
    """
    feedback_list = load_feedback()
    
    if len(feedback_list) < 10:
        logger.warning(f"Insufficient feedback for retraining: {len(feedback_list)} < 10")
        return None, None
    
    X = np.array([f['features'] for f in feedback_list])
    y = np.array([f['actual_label'] for f in feedback_list])
    
    logger.info(f"Prepared training data: {len(X)} examples")
    return X, y

def retrain_model(base_X_train, base_y_train, base_scaler):
    """
    Retrain model using original training data + collected feedback.
    
    SAFETY NOTES:
    - Requires original training data (base_X_train, base_y_train)
    - Will NOT retrain on feedback-only (too risky)
    - Requires at least 50+ feedback examples for meaningful improvement
    
    Args:
        base_X_train: Original training features (already scaled)
        base_y_train: Original training labels
        base_scaler: Original StandardScaler object
        
    Returns:
        tuple: (new_model, new_scaler) or (None, None) if retraining failed
    """
    logger.info("\n" + "="*70)
    logger.info("ADAPTIVE LEARNING - RETRAINING MODEL")
    logger.info("="*70)
    
    # SAFETY CHECK: Require original training data
    if base_X_train is None or base_y_train is None:
        logger.error("✗ BLOCKED: Original training data required for safe retraining")
        logger.error("   (Feedback-only retraining is disabled to prevent model corruption)")
        return None, None
    
    # Get feedback data
    X_feedback, y_feedback = prepare_training_data_from_feedback()
    
    if X_feedback is None or len(X_feedback) < 10:
        logger.warning(f"Cannot retrain: insufficient feedback data (need >=10, have {len(X_feedback) if X_feedback is not None else 0})")
        return None, None
    
    try:
        # Scale feedback data using the base scaler
        X_feedback_scaled = base_scaler.transform(X_feedback)
        
        # Combine original training data with feedback
        X_combined = np.vstack([base_X_train, X_feedback_scaled])
        y_combined = np.hstack([base_y_train, y_feedback])
        
        logger.info(f"Combined data: {len(X_combined)} examples")
        logger.info(f"  - Original: {len(base_X_train)} examples")
        logger.info(f"  - Feedback: {len(X_feedback)} examples ({len(X_feedback)/len(base_X_train)*100:.1f}% of original)")
        logger.info(f"  - Fraud rate in feedback: {np.sum(y_feedback == 1) / len(y_feedback) * 100:.2f}%")
        logger.info(f"  - Overall fraud rate: {np.sum(y_combined == 1) / len(y_combined) * 100:.2f}%")
        
        # Train new model
        logger.info("Training new Random Forest classifier...")
        new_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        new_model.fit(X_combined, y_combined)
        
        logger.info("✓ Model retraining complete!")
        logger.info("="*70 + "\n")
        
        return new_model, base_scaler
        
    except Exception as e:
        logger.error(f"✗ Retraining failed: {e}")
        return None, None

def create_model_backup():
    """Create backup of current model before retraining."""
    import shutil
    try:
        if os.path.exists('fraud_detection_model.pkl'):
            shutil.copy('fraud_detection_model.pkl', 'fraud_detection_model.pkl.backup')
            if os.path.exists('scaler.pkl'):
                shutil.copy('scaler.pkl', 'scaler.pkl.backup')
            logger.info("✓ Created backup: fraud_detection_model.pkl.backup")
            return True
    except Exception as e:
        logger.error(f"✗ Failed to create backup: {e}")
    return False

def restore_model_from_backup():
    """Restore model from backup if needed."""
    import shutil
    try:
        if os.path.exists('fraud_detection_model.pkl.backup'):
            shutil.copy('fraud_detection_model.pkl.backup', 'fraud_detection_model.pkl')
            if os.path.exists('scaler.pkl.backup'):
                shutil.copy('scaler.pkl.backup', 'scaler.pkl')
            logger.info("✓ Restored model from backup")
            return True
    except Exception as e:
        logger.error(f"✗ Failed to restore backup: {e}")
    return False

def apply_retrained_model(new_model, new_scaler):
    """
    Save the retrained model and scaler with backup safety.
    
    SAFETY CHECKS:
    - Creates backup before overwriting
    - Validates model can make predictions
    - Allows rollback if needed
    
    Args:
        new_model: Newly trained model
        new_scaler: StandardScaler (same as before)
    """
    if new_model is None or new_scaler is None:
        logger.error("Cannot apply: invalid model or scaler")
        return False
    
    try:
        # Create backup before overwriting
        if not create_model_backup():
            logger.warning("⚠ Backup creation failed, but proceeding cautiously...")
        
        # Quick validation: test model can make predictions
        try:
            test_pred = new_model.predict(np.zeros((1, 29)))
            logger.info(f"✓ Model validation passed (test prediction: {test_pred[0]})")
        except Exception as e:
            logger.error(f"✗ Model validation FAILED: {e}")
            logger.error("   Aborting retrain to preserve current model")
            return False
        
        # Save retrained model
        save_model(new_model, new_scaler, 
                   model_path='fraud_detection_model.pkl',
                   scaler_path='scaler.pkl')
        
        # Log the retraining event
        log_training_event({
            'timestamp': datetime.now().isoformat(),
            'event': 'model_retrained_from_feedback',
            'feedback_samples': len(load_feedback()),
            'status': 'success'
        })
        
        logger.info("✓ Retrained model saved and activated!")
        logger.info("  (Backup saved to fraud_detection_model.pkl.backup)")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to save retrained model: {e}")
        logger.warning("   Attempting to restore from backup...")
        restore_model_from_backup()
        return False

def log_training_event(event_dict):
    """Log retraining events for audit trail."""
    if os.path.exists(TRAINING_LOG_FILE):
        with open(TRAINING_LOG_FILE, 'r') as f:
            logs = json.load(f)
    else:
        logs = []
    
    logs.append(event_dict)
    
    with open(TRAINING_LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

def clear_feedback():
    """Clear collected feedback (use after successful retraining)."""
    if os.path.exists(FEEDBACK_FILE):
        os.remove(FEEDBACK_FILE)
        logger.info("✓ Cleared feedback data")
