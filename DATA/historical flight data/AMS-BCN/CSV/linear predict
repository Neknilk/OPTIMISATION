import os
import glob
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Load and combine all CSV files
folder_path = "C:/Users/jayva/Documents/GitHub/OPTIMISATION/DATA/historical flight data/AMS-BCN/CSV/cleaned"
all_files = glob.glob(os.path.join(folder_path, "*.csv"))
combined_data = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)

# Drop rows with missing values in the selected fuel flow columns
combined_data = combined_data.dropna(subset=['SELECTED FUEL FLOW #2 (KG)', 'SELECTED FUEL FLOW #1 (KG)'])

# Create the feature matrix X and target variable y
X = combined_data[['TOTAL AIR TEMP']]
y = combined_data['SELECTED FUEL FLOW #2 (KG)'] + combined_data['SELECTED FUEL FLOW #1 (KG)']

# Create and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Save the trained model to a file
model_filename = "C:/Users/jayva/Documents/GitHub/OPTIMISATION/CODE/FF model.joblib"
joblib.dump(model, model_filename)

print(f"Model saved to {model_filename}")

# Function to predict fuel flow based on temperature
def predict_fuel_flow(temperature):
    input_data = [[temperature]]
    fuel_flow_prediction = model.predict(input_data)
    return fuel_flow_prediction[0]

# Input value for temperature
input_temperature = float(input("Enter the temperature: "))

# Predict fuel flow based on user input
predicted_fuel_flow = predict_fuel_flow(input_temperature)
print(f'Predicted Fuel Flow: {predicted_fuel_flow:.2f} KG')
