# Necessary imports
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

# Load the csv file 
# 0 is male and 1 is female for gender
coffee_data = pd.read_csv("model/coffee-prefrences.csv")

# Split the data into features (X) and labels (y)
# X contains "gender" and "age"
# y contains "type_of_coffee"
X = coffee_data.drop(columns="type_of_coffee")
y = coffee_data["type_of_coffee"]

# Split the data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "model/coffee_model.joblib")
