import os
import pandas as pd

# Input and output folder paths
input_folder = r'C:\Users\jayva\Documents\GitHub\OPTIMISATION\DATA\historical flight data\AMS-BCN\CSV\phased'
output_folder = r'C:\Users\jayva\Documents\GitHub\OPTIMISATION\DATA\historical flight data\AMS-BCN\CSV\cleaned'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Iterate over CSV files in the input folder
for flight_number in range(1, 8):
    input_file_path = os.path.join(input_folder, f'Flight {flight_number}_with_phases_and_changes.csv')
    output_file_path = os.path.join(output_folder, f'Flight {flight_number}.csv')

    # Read the CSV file
    df = pd.read_csv(input_file_path)

    # Filter rows where "Flight Phases" column contains "cruise"
    df_cruise = df[df['Flight Phases'].str.contains('cruise', case=False, na=False)]

    # Save the cleaned data to the output folder
    df_cruise.to_csv(output_file_path, index=False)

    print(f'Flight {flight_number} cleaned and saved to {output_file_path}')
