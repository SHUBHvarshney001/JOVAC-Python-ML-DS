# model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the data
data = pd.read_csv("DATA.csv")

# Data Preprocessing
data['Car'] = data['Car'] + ' ' + data['Model']  # Combine 'Car' and 'Model'
data.drop(data.columns[1], axis=1, inplace=True)  # Drop the 'Model' column
data['Volume*Weight'] = data['Volume'] * data['Weight']

# Defining Features and Target Variable
X = data[['Volume', 'Weight', 'Volume*Weight']]
y = data['CO2']

# Splitting the Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train Linear Regression model
linear_regr = LinearRegression()
linear_regr.fit(X_train, y_train)

def predict_co2(volume, weight):
    """Predict CO2 emission based on volume and weight using the trained Linear Regression model."""
    volume_weight = volume * weight
    input_features = np.array([[volume, weight, volume_weight]])
    co2_pred = linear_regr.predict(input_features)
    return co2_pred[0]
