"""
model.py
Train, evaluate, and save the fraud detection model.
Handles model training with Random Forest and evaluation metrics.
"""

import logging
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(X_train, y_train, n_estimators=100):
    """
    Train Random Forest classifier.
    
    Args:
        X_train (np.ndarray): Training features (scaled)
        y_train (np.ndarray): Training labels
        n_estimators (int): Number of trees in forest
        
    Returns:
        RandomForestClassifier: Trained model
    """
    logging.info(f"Training Random Forest with {n_estimators} estimators...")
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Model training complete!")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Args:
        model: Trained model
        X_test (np.ndarray): Test features (scaled)
        y_test (np.ndarray): Test labels
        
    Returns:
        dict: Evaluation metrics
    """
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logging.info("=" * 60)
    logging.info("MODEL EVALUATION RESULTS")
    logging.info("=" * 60)
    logging.info(f"Accuracy:  {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall:    {recall:.4f}")
    logging.info(f"F1-Score:  {f1:.4f}")
    logging.info("\nConfusion Matrix:")
    logging.info(confusion_matrix(y_test, y_pred))
    logging.info("\nClassification Report:")
    logging.info(classification_report(y_test, y_pred, zero_division=0))
    logging.info("=" * 60)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'predictions': y_pred
    }

def save_model(model, scaler, model_path='fraud_detection_model.pkl', scaler_path='scaler.pkl'):
    """
    Save trained model and scaler to disk.
    
    Args:
        model: Trained model
        scaler: StandardScaler object
        model_path (str): Path to save model
        scaler_path (str): Path to save scaler
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    logging.info(f"Model saved to {model_path}")
    logging.info(f"Scaler saved to {scaler_path}")

def load_model(model_path='fraud_detection_model.pkl', scaler_path='scaler.pkl'):
    """
    Load trained model and scaler from disk.
    
    Args:
        model_path (str): Path to model file
        scaler_path (str): Path to scaler file
        
    Returns:
        tuple: (model, scaler)
    """
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    logging.info(f"Model loaded from {model_path}")
    logging.info(f"Scaler loaded from {scaler_path}")
    return model, scaler

def predict_transaction(model, scaler, features):
    """
    Make prediction for a single transaction.
    
    Args:
        model: Trained model
        scaler: StandardScaler object
        features (array-like): Feature values (must match training features)
        
    Returns:
        tuple: (prediction, probability)
            prediction: 0 (normal) or 1 (fraud)
            probability: Confidence score
    """
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    return prediction, probability
