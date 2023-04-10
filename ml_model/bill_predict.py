#!/usr/bin/env python3
"""Creating a prediction model"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder  # Used to convert text to 0s and 1s
from sklearn.compose import ColumnTransformer  # Used to convert text to 0s and 1s
from sklearn.model_selection import train_test_split
import pickle   # Used to save the trained model


# # Load the csv file
# data = pd.read_csv("Electricity_bill.csv")
# # print(data)

# # Convert appliance names to integers using LabelEncoder
# le = LabelEncoder()
# data['Appliance'] = le.fit_transform(data['Appliance'])

# # Define X and y for the model
# X = data[['Appliance', 'Wattage', 'Time', 'Number of Days Used']].values
# y = data[['Daily_Consumption', 'Monthly_Consumption']].values

# # Use OneHotEncoder to create dummy variable columns for Appliance
# ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = ct.fit_transform(X)
# X = X[:, 1:]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Create a linear regression model and train it using the training data
# model = LinearRegression()
# model.fit(X_train, y_train)

# # Print the accuracy of the model on the testing data
# print(f"Model accuracy on testing data: {model.score(X_test, y_test)}")

# # Save the trained model
# with open('electricity_bill_model.pkl', 'wb') as f:
#     pickle.dump(model, f)

# # Load the saved model
# with open('electricity_bill_model.pkl', 'rb') as f:
#     loaded_model = pickle.load(f)

# # Make a prediction using the loaded model
# prediction = loaded_model.predict([[1, 0, 0, 1000, 2, 30, 1]])
# print(f"Prediction: {prediction}")

# Define a function that accepts input from the frontend API and returns the predicted values
# def predict(input_data):
#     # Load the saved model
#     with open('electricity_bill_model.pkl', 'rb') as f:
#         loaded_model = pickle.load(f)

#     # Convert the input data to a pandas DataFrame
#     input_df = pd.DataFrame([input_data])

#     # Convert appliance names to integers using LabelEncoder
#     le = LabelEncoder()

#     # Convert appliance names to integers using LabelEncoder
#     input_df['Appliance'] = le.transform(input_df['Appliance'])

#     # Define X for the model
#     X = input_df[['Appliance', 'Wattage', 'Time', 'Number of Days Used']].values

#     # Use OneHotEncoder to create dummy variable columns for Appliance
#     ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#     X = ct.fit_transform(X)
#     X = X[:, 1:]

#     # Make a prediction using the loaded model
#     prediction = loaded_model.predict(X)

#     return prediction


# Load the csv file
data = pd.read_csv("Electricity.csv")

# Convert appliance names to integers using LabelEncoder
le = LabelEncoder()
data['Appliance'] = le.fit_transform(data['Appliance'])

# Define X and y for the model
X = data[['Appliance', 'Wats', 'Time', 'Days']].astype(float).values
y = data[['D_Consumption', 'M_Consumption']].values
print(X)
print(y)
print(data.columns)

# Use OneHotEncoder to create dummy variable columns for Appliance
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

model = LinearRegression()
model.fit(X, y)
print(model.coef_)
print(model.intercept_)
print(f"Model accuracy on testing data: {model.score(X, y)}")

# Select the first binary variable as input to the model
# X = X[:, 0]

# # Reshape X to a 2D array
# X = X.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create a linear regression model and train it using the training data
model.fit(X_train, y_train)

# Save the trained model
with open('electricity-bill', 'wb') as f:
    pickle.dump(model, f)

# Load the saved model
with open('electricity-bill', 'rb') as f:
    loaded_model = pickle.load(f)

# Make a prediction using the loaded model
# 0: A binary variable indicating whether the first appliance in the dataset is being used (0 or 1)
# 1000: Wattage of the appliance being used
# 2: Time in hours the appliance is being used
# 30: Number of days the appliance is being used
# 1: A binary variable indicating the type of user (1 for residential, 0 for commercial)
prediction = loaded_model.predict([[1, 0, 0, 0, 1000, 2, 30]])
print(f"Prediction: {prediction}")

# # Load data from API
# api_data = pd.read_json("http://localhost:8000/predict-bille")

# # Convert appliance names to integers using LabelEncoder
# le = LabelEncoder()
# api_data['Appliance'] = le.fit_transform(api_data['Appliance'])

# # Define X and y for the model
# X = api_data[['Appliance', 'Wattage', 'Time', 'Number of Days Used']].values
# y = api_data[['Daily_Consumption', 'Monthly_Consumption']].values

# # Use OneHotEncoder to create dummy variable columns for Appliance
# ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
# X = ct.fit_transform(X)
# X = X[:, 1:]

# # Create a linear regression model and train it using the new data
# model = LinearRegression()
# model.fit(X, y)

# # Save the trained model
# with open('bill_model.pkl', 'wb') as f:
#     pickle.dump(model, f)