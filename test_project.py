"""
test_project.py
Verification script to test that the entire project is working correctly.

Usage:
    python test_project.py

This script will:
1. Check if all required files exist
2. Verify model files can be loaded
3. Test a sample prediction
4. Display system information
"""

import os
import sys
from pathlib import Path

print("\n" + "=" * 70)
print("CREDIT CARD FRAUD DETECTION - PROJECT VERIFICATION")
print("=" * 70 + "\n")

# Color codes for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_file(file_path, description):
    """Check if a file exists."""
    if os.path.exists(file_path):
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"{GREEN}✓{RESET} {description:<40} | Size: {size_mb:.2f} MB")
        return True
    else:
        print(f"{RED}✗{RESET} {description:<40} | NOT FOUND")
        return False

def check_module(module_name):
    """Check if a Python module can be imported."""
    try:
        __import__(module_name)
        print(f"{GREEN}✓{RESET} {module_name}")
        return True
    except ImportError:
        print(f"{RED}✗{RESET} {module_name}")
        return False

# Track results
all_good = True

# ===== CHECK 1: Required Files =====
print("\n[CHECK 1] Required Project Files")
print("-" * 70)

files_to_check = [
    ('creditcard.csv', 'Original dataset'),
    ('fraud_detection_model.pkl', 'Trained model'),
    ('scaler.pkl', 'Feature scaler'),
    ('main.py', 'Main entry point'),
    ('data_loader.py', 'Data preprocessing module'),
    ('model.py', 'Model training & prediction'),
    ('app.py', 'Flask API'),
    ('README.md', 'Documentation'),
]

for file_name, description in files_to_check:
    if not check_file(file_name, description):
        all_good = False

# ===== CHECK 2: Required Python Libraries =====
print("\n[CHECK 2] Required Libraries")
print("-" * 70)

libraries = [
    'pandas',
    'numpy',
    'sklearn',
    'joblib',
    'flask',
]

missing_libs = []
for lib in libraries:
    if not check_module(lib):
        missing_libs.append(lib)
        all_good = False

# ===== CHECK 3: Load Model =====
print("\n[CHECK 3] Model Loading Test")
print("-" * 70)

try:
    from model import load_model
    model, scaler = load_model()
    print(f"{GREEN}✓{RESET} Model loaded successfully")
    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Scaler type: {type(scaler).__name__}")
except Exception as e:
    print(f"{RED}✗{RESET} Failed to load model: {e}")
    all_good = False
    model = None
    scaler = None

# ===== CHECK 4: Sample Prediction =====
print("\n[CHECK 4] Sample Prediction Test")
print("-" * 70)

if model and scaler:
    try:
        from model import predict_transaction
        import numpy as np
        
        # Create a sample transaction (all features with dummy values)
        sample_features = np.array([
            -1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10,
            0.36, 0.09, -0.55, -0.62, -0.99, -0.31, 1.47, -0.47,
            0.21, 0.03, 0.40, 0.25, -0.02, 0.28, -0.11, 0.07,
            0.13, -0.19, 0.13, -0.02, 50.0
        ])
        
        prediction, probabilities = predict_transaction(model, scaler, sample_features)
        
        print(f"{GREEN}✓{RESET} Prediction successful")
        print(f"  - Input: 29 features (V1-V28 + Amount)")
        print(f"  - Prediction: {prediction} {'(FRAUD)' if prediction == 1 else '(NORMAL)'}")
        print(f"  - Normal probability: {probabilities[0]:.4f}")
        print(f"  - Fraud probability: {probabilities[1]:.4f}")
        
    except Exception as e:
        print(f"{RED}✗{RESET} Prediction test failed: {e}")
        all_good = False
else:
    print(f"{RED}✗{RESET} Cannot test prediction (model not loaded)")
    all_good = False

# ===== CHECK 5: Data Loading =====
print("\n[CHECK 5] Data Loading Test")
print("-" * 70)

try:
    from data_loader import load_data, preprocess_data
    
    data = load_data('creditcard.csv')
    print(f"{GREEN}✓{RESET} Dataset loaded successfully")
    print(f"  - Shape: {data.shape}")
    print(f"  - Rows: {data.shape[0]:,}")
    print(f"  - Columns: {data.shape[1]}")
    
    X, y = preprocess_data(data)
    print(f"{GREEN}✓{RESET} Data preprocessing successful")
    print(f"  - Features (X): {X.shape}")
    print(f"  - Labels (y): {y.shape}")
    print(f"  - Normal: {(y==0).sum():,}, Fraud: {(y==1).sum():,}")
    
except Exception as e:
    print(f"{RED}✗{RESET} Data loading test failed: {e}")
    all_good = False

# ===== CHECK 6: Flask App Import =====
print("\n[CHECK 6] Flask Application Test")
print("-" * 70)

try:
    from app import app
    print(f"{GREEN}✓{RESET} Flask app imports successfully")
    print(f"  - App name: {app.name}")
    print(f"  - Debug mode: {app.debug}")
except Exception as e:
    print(f"{RED}✗{RESET} Flask app import failed: {e}")
    all_good = False

# ===== SUMMARY =====
print("\n" + "=" * 70)
print("TEST SUMMARY")
print("=" * 70 + "\n")

if all_good:
    print(f"{GREEN}✓ ALL CHECKS PASSED!{RESET}")
    print("\nYour project is ready to use!")
    print("\nNext steps:")
    print("  1. Train model:      python main.py")
    print("  2. Run Flask API:    python app.py")
    print("  3. Read docs:        README.md or QUICK_START.md")
    print("\n" + "=" * 70 + "\n")
    sys.exit(0)
else:
    print(f"{RED}✗ SOME CHECKS FAILED{RESET}")
    print(f"\nIssues to fix:")
    if missing_libs:
        print(f"  - Install missing libraries: pip install {' '.join(missing_libs)}")
    print(f"  - Check that all files are in the correct directory")
    print(f"  - Run 'python main.py' to train if model files are missing")
    print("\n" + "=" * 70 + "\n")
    sys.exit(1)
