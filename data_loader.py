"""
data_loader.py
Load and preprocess credit card fraud dataset.
Handles data preprocessing: feature selection, scaling, train-test split.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(filepath='creditcard.csv'):
    """
    Load credit card dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    try:
        data = pd.read_csv(filepath)
        logging.info(f"Loaded data from {filepath}. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        logging.error(f"File {filepath} not found!")
        raise

def preprocess_data(data):
    """
    Preprocess data for model training.
    Separates features and labels. Uses V1-V28 (PCA features) + Amount.
    Note: Time column is in seconds in the original dataset.
    
    Args:
        data (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (X, y) - Features and labels
    """
    # Select features: All V columns (V1-V28) and Amount
    feature_columns = [col for col in data.columns if col.startswith('V')] + ['Amount']
    
    X = data[feature_columns].values
    y = data['Class'].values
    
    logging.info(f"Selected {len(feature_columns)} features: V1-V28 + Amount")
    logging.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logging.info(f"Class distribution - Normal: {np.sum(y==0)}, Fraud: {np.sum(y==1)}")
    
    return X, y

def split_and_scale_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train/test sets and scale features.
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Labels
        test_size (float): Proportion of test set
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
    """
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logging.info(f"Train set size: {X_train_scaled.shape[0]}, Test set size: {X_test_scaled.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def prepare_data(filepath='creditcard.csv', test_size=0.2):
    """
    Complete pipeline: Load → Preprocess → Split → Scale
    
    Args:
        filepath (str): Path to dataset
        test_size (float): Test set proportion
        
    Returns:
        dict: Dictionary containing all prepared data and scaler
    """
    data = load_data(filepath)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test, scaler = split_and_scale_data(X, y, test_size=test_size)
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'data': data
    }
