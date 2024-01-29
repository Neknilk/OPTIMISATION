import os
import pandas as pd

# Path to the folder containing your CSV files
input_folder = r'C:\Users\jayva\Documents\GitHub\OPTIMISATION\DATA\historical flight data\AMS-BCN\CSV\cleaned'
output_folder = r'C:\Users\jayva\Documents\GitHub\OPTIMISATION\DATA\historical flight data\AMS-BCN\CSV\cleaned'

# Get a list of all CSV files in the input folder
csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

# Iterate through each CSV file
for csv_file in csv_files:
    input_path = os.path.join(input_folder, csv_file)
    output_path = os.path.join(output_folder, csv_file)

    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(input_path)

    # Filter rows based on the condition (value in '3d Altitude Ft' column is lower than 20000)
    df = df[df['[3d Altitude Ft]'] >= 20000]

    # Drop rows where any of the specified columns has missing values
    df.dropna(subset=['SELECTED FUEL FLOW #1 (KG)', 'SELECTED FUEL FLOW #2 (KG)', 'TRUE AIRSPEED (derived)'], how='any', inplace=True)

    # Save the cleaned DataFrame to the output folder
    df.to_csv(output_path, index=False)

    print(f"Processed {csv_file}")

print("All files processed.")
