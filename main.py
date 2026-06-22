import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
insurance_dataset = pd.read_csv("insurance.csv")

# Encode categorical columns
insurance_dataset.replace({
    'sex': {'male': 0, 'female': 1},
    'smoker': {'yes': 1, 'no': 0},
    'region': {
        'southwest': 0,
        'southeast': 1,
        'northwest': 2,
        'northeast': 3
    }
}, inplace=True)

# Split features and target
X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2
)

# Train model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Training score
training_data_prediction = regressor.predict(X_train)
r2_train = r2_score(Y_train, training_data_prediction)

# Testing score
test_data_prediction = regressor.predict(X_test)
r2_test = r2_score(Y_test, test_data_prediction)

print("Training R² Score =", r2_train)
print("Testing R² Score =", r2_test)

# Prediction system
input_data = (31, 0, 25.74, 0, 0, 1)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = regressor.predict(input_data_reshaped)

print("Predicted Insurance Cost = $", prediction[0])
