from data_loader import prepare_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from model import save_model
import logging
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

print('Loading data...')
data_dict = prepare_data(filepath='creditcard.csv', test_size=0.2)
X_train, y_train = data_dict['X_train'], data_dict['y_train']
X_test, y_test = data_dict['X_test'], data_dict['y_test']
scaler = data_dict['scaler']

print('Training model (this takes ~2 minutes)...')
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=0)
model.fit(X_train, y_train)

print('Evaluating...')
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
print(f'Accuracy: {acc:.4f}, Fraud Recall: {rec:.4f}')

print('Saving...')
save_model(model, scaler)
print('✓ Model restored!')
