# pip install pandas
# pip install sklearn
# pip install Xgboost



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb

# Load the dataset

df = pd.read_csv('fraud_indicators.csv')

# Data preprocessing
X = df.drop('is_fraud', axis=1) 
y = df['is_fraud']  

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features using StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions on the test set
rf_predictions = rf_classifier.predict(X_test)

# Evaluate the Random Forest model
print("Random Forest Classifier Results:")
print(confusion_matrix(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions))

# XGBoost Classifier
xgb_classifier = xgb.XGBClassifier(objective='binary:logistic', use_label_encoder=False, random_state=42)
xgb_classifier.fit(X_train, y_train)

# Predictions on the test set
xgb_predictions = xgb_classifier.predict(X_test)

# Evaluate the XGBoost model
print("\nXGBoost Classifier Results:")
print(confusion_matrix(y_test, xgb_predictions))
print(classification_report(y_test, xgb_predictions))
