# Necessary imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load data
coffee_data = pd.read_csv("model/coffee-prefrences.csv")

# Features and target
X = coffee_data.drop(columns="type_of_coffee")
y = coffee_data["type_of_coffee"]

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Load trained model
model = joblib.load("model/coffee_model.joblib")

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test data: {accuracy:.2%}")
