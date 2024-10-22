from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import random  # Importing the random module

app = Flask(__name__)

# Suggestions dictionary
suggestions = {
    'car model 1': 'Consider using public transportation or carpooling to reduce emissions.',
    'car model 2': 'Regular maintenance can improve fuel efficiency. Consider an electric vehicle for lower emissions.',
    'car model 3': 'Driving at moderate speeds can reduce fuel consumption significantly.',
    'car model 4': 'Keep your tires properly inflated to improve fuel efficiency.',
    'car model 5': 'Using cruise control on highways can help save fuel.',
    'car model 6': 'Consider a hybrid or electric vehicle for lower emissions.',
    'car model 7': 'Avoid excessive idling; turn off your engine when parked.',
    'car model 8': 'Use air conditioning sparingly to improve fuel economy.',
}

# Load the data
data = pd.read_csv('DATA.csv')

# Data Preprocessing
data['Car'] = data['Car'] + ' ' + data['Model']  # Combine 'Car' and 'Model'
data.drop(data.columns[1], axis=1, inplace=True)  # Drop the 'Model' column
data['Volume*Weight'] = data['Volume'] * data['Weight']

# Train the Linear Regression Model
X = data[['Volume', 'Weight', 'Volume*Weight']]
y = data['CO2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

linear_regr = LinearRegression()
linear_regr.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    car = request.form['car'].strip().lower()  # Normalize input
    print(f"Received car input: '{car}'")  # Debug statement
    volume = float(request.form['volume'])
    weight = float(request.form['weight'])
    volume_weight = volume * weight

    # Prepare the input for prediction
    input_features = np.array([[volume, weight, volume_weight]])
    co2_pred = linear_regr.predict(input_features)

    # Select a random suggestion from the suggestions
    random_suggestion = random.choice(list(suggestions.values()))  # Randomly select a suggestion
    print(f"Random suggestion: '{random_suggestion}'")  # Debug statement

    return render_template('index.html', 
                           prediction_text=f'Predicted CO2 Emission: {co2_pred[0]:.2f} g/km', 
                           suggestion_text=random_suggestion)

if __name__ == "__main__":
    app.run(debug=True)
