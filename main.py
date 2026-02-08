"""
main.py
MAIN ENTRY POINT for Credit Card Fraud Detection Project.

This script orchestrates the complete machine learning pipeline:
1. Load data from creditcard.csv
2. Preprocess features (select V1-V28 + Amount)
3. Split data into train/test sets
4. Scale features using StandardScaler
5. Train Random Forest classifier
6. Evaluate model performance
7. Save model and scaler for later use

Usage:
    python main.py              # Train model and display results
    python main.py --app        # Train model, then run Flask web app

After training, you can:
- Use 'app.py' to run the Flask API for predictions
- Use the custom functions in 'model.py' for advanced operations
"""

import sys
import logging
import pickle
from data_loader import prepare_data
from model import train_model, evaluate_model, save_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main pipeline: Load data → Train model → Evaluate → Save
    """
    logger.info("=" * 70)
    logger.info("CREDIT CARD FRAUD DETECTION - MAIN PIPELINE")
    logger.info("=" * 70)
    
    # Step 1: Load and prepare data
    logger.info("\n[STEP 1/4] Loading and preparing data...")
    try:
        data_dict = prepare_data(filepath='creditcard.csv', test_size=0.2)
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        scaler = data_dict['scaler']
        logger.info("✓ Data preparation complete!")
    except Exception as e:
        logger.error(f"✗ Failed to load data: {e}")
        return False
    
    # Step 2: Train model
    logger.info("\n[STEP 2/4] Training Random Forest model...")
    try:
        model = train_model(X_train, y_train, n_estimators=100)
        logger.info("✓ Model training complete!")
    except Exception as e:
        logger.error(f"✗ Failed to train model: {e}")
        return False
    
    # Step 3: Evaluate model
    logger.info("\n[STEP 3/4] Evaluating model performance...")
    try:
        metrics = evaluate_model(model, X_test, y_test)
        logger.info("✓ Model evaluation complete!")
    except Exception as e:
        logger.error(f"✗ Failed to evaluate model: {e}")
        return False
    
    # Step 4: Save model
    logger.info("\n[STEP 4/4] Saving model and scaler...")
    try:
        save_model(model, scaler)
        logger.info("✓ Model and scaler saved successfully!")
        
        # Save training data for adaptive learning
        training_data = {
            'X_train': X_train,
            'y_train': y_train,
            'scaler': scaler
        }
        with open('training_data.pkl', 'wb') as f:
            pickle.dump(training_data, f)
        logger.info("✓ Training data saved for adaptive learning!")
        
    except Exception as e:
        logger.error(f"✗ Failed to save model: {e}")
        return False
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"Final Model Performance:")
    logger.info(f"  • Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  • Precision: {metrics['precision']:.4f}")
    logger.info(f"  • Recall:    {metrics['recall']:.4f}")
    logger.info(f"  • F1-Score:  {metrics['f1_score']:.4f}")
    logger.info("\nModel files saved:")
    logger.info("  • fraud_detection_model.pkl")
    logger.info("  • scaler.pkl")
    logger.info("  • training_data.pkl (for adaptive learning)")
    logger.info("\nAdaptive Learning:")
    logger.info("  Users can provide feedback on predictions")
    logger.info("  Admin can trigger model retraining with collected feedback")
    logger.info("  Training data is automatically combined with feedback for improvement")
    logger.info("\nNext steps:")
    logger.info("  1. To run the Flask API: python app.py")
    logger.info("  2. To make predictions programmatically: from model import load_model, predict_transaction")
    logger.info("=" * 70 + "\n")
    
    return True

if __name__ == '__main__':
    # Run main pipeline
    success = main()
    
    # Optionally run Flask app if --app flag is provided
    if '--app' in sys.argv:
        logger.info("\nStarting Flask app...")
        from app import app
        app.run(debug=True)
    
    sys.exit(0 if success else 1)
