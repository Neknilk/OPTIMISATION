import os
import pandas as pd

# Specify the folder containing the CSV files
folder_path = r'PSO\DATA\historical flight data\AMS-BCN\CSV\RAW'

# Loop through each file in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(folder_path, filename)

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path, low_memory=False)

        # Filter rows where [3d Latitude] and [3d Longitude] are not equal to 0
        df = df[(df['[3d Latitude]'] != 0) | (df['[3d Longitude]'] != 0)]

        # Save the modified DataFrame back to the CSV file
        df.to_csv(file_path, index=False)

        print(f"Processed file: {filename}")
