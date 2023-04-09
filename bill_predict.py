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


# Load the csv file
data = pd.read_csv("Electricity_bill.csv")
# print(data)

# Converting text to 0s and 1s using "One-Hot Encoding"
# dummies = pd.get_dummies(data, columns=['Appliance'])
# dummies = dummies.astype(int)
# # print(dummies)

# # Merge the dummies data frame with the original data frame
# merged = pd.concat([data, dummies], axis='columns')
# # print(merged)

# # Drop the text and one of the 0s and 1s in the merged data
# final_data = merged.drop(['Appliance', 'Appliance_Iron'], axis='columns')
# print(final_data)

# # Create a linear regression model
# model = LinearRegression()
# # Put data into X and y variables
# X = final_data.drop(['Daily_Consumption', 'Monthly_Consumption'], axis='columns')
# y = final_data[['Daily_Consumption', 'Monthly_Consumption']]


# # Trainining the model
# model.fit(X,y)

# # Knowing the accuracy of the model
# print(model.score(X,y))
# # Make a prediction based on input [1, 1000, 2, 30, 1, 0]
# prediction = model.predict([[1000, 2, 30, 1, 1, 0, 0, 0, 1, 1]])

# # Print the prediction
# print(prediction)

# Create a linear regression model
model = LinearRegression()

# Using one-hot encoding to achieve the same result
# Create a label encoder object
le = LabelEncoder()
# Create a new data frame
datale = data
# Fit and transform the data frame
datale.Appliance = le.fit_transform(datale.Appliance)   # Converting appliance names to integers
print(datale)
# X = datale.drop(['Daily_Consumption', 'Monthly_Consumption'], axis='columns')
X = datale[['Appliance', 'Wattage', 'Time', 'Number of Days Used']].values
print(X)
y = datale[['Daily_Consumption', 'Monthly_Consumption']].values
print(y)

#  Create dummy variable columns
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = ct.fit_transform(X)
# Drop the first column
X = X[:, 1:]
print(X)

# Trainining the model using the new data frame
model.fit(X,y)
# Predicting the result
prediction = model.predict([[1, 0, 0, 1000, 2, 30, 1]])
prediction1 = model.predict([[2, 1, 3, 200, 2, 20, 1]])
print(prediction)
print(prediction1)

# Print the accuracy of the model
print(model.score(X,y))

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Train the model using the training data
model.fit(X_train, y_train)

# Predict the result using the testing data
y_pred = model.predict(X_test)
print(y_pred)

# Print the accuracy of the model
print(model.score(X_test, y_test))

# Save the model
with open('electricity_bill_model', 'wb') as f:
    pickle.dump(model, f)

# Load the model
with open('electricity_bill_model', 'rb') as f:
    mp = pickle.load(f)

# Make a prediction using the loaded model
prediction = mp.predict([[1, 0, 0, 1000, 2, 30, 1]])
print(prediction)

# Create linear regression object
# reg = linear_model.LinearRegression()
# # Use fit() to train the model and create the data frame
# # Independent variables go into the data frame and the other is the target variable
# reg.fit(data[['Appliance', 'Wattage', 'Time', 'Number of Days Used']], data.Daily_Consumption, data.Monthly_Consumption)
# print(f"The coeficients are: {reg.coef_}")
# print(f"The interceptor is: {reg.intercept_}")

# #Calculate the predicted prices
# print(reg.predict([[]]))