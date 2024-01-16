import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch
import joblib

# Sample data (replace with your actual data)
data = {
    'ActivityType': ['Transportation', 'Food', 'Energy Use', 'Waste Management'],
    'Quantity': [50, 3, 100, 20],
    'CarbonFootprint': [30, 5, 50, 10]
}

df = pd.DataFrame(data)

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['ActivityType'])

# Split the data into features (X) and target variable (y)
X = df.drop('CarbonFootprint', axis=1)
y = df['CarbonFootprint']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the model-building function
def build_model(hp):
    model = keras.Sequential()
    model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))
    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i), min_value=8, max_value=64, step=8),
                               activation='relu'))
    model.add(layers.Dense(1, activation='linear'))

    model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])
    return model

# Set up the hyperparameter tuner
tuner = RandomSearch(build_model,
                     objective='val_mean_squared_error',
                     max_trials=5,
                     directory='keras_tuner',
                     project_name='carbon_footprint')

# Search for the best hyperparameter configuration
tuner.search(X_train, y_train, epochs=50, validation_split=0.1)

# Get the best model
best_model = tuner.get_best_models(num_models=1)[0]

# Save the model to a file
joblib.dump(best_model, 'saved_model.joblib')

# Evaluate the best model on the test set
mse = best_model.evaluate(X_test, y_test)[1]
print(f"Mean Squared Error on Test Set: {mse:.2f}")

# Make predictions on new data
new_data = pd.DataFrame({
    'Quantity': [50],
    'ActivityType_Transportation': [0],
    'ActivityType_Food': [1],
    'ActivityType_Energy Use': [0],
    'ActivityType_Waste Management': [0]
})  # Replace with your own data

# Ensure the column order is the same as the one-hot encoded training data
new_data = new_data[X.columns]

new_data = scaler.transform(new_data)
predictions = best_model.predict(new_data)
print(f"Predicted Carbon Footprint: {predictions[0][0]:.2f} kgCO2")
