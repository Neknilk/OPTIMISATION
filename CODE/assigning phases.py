import os
import pandas as pd

# Path to the directory containing CSV files
input_directory_path = r'C:\Users\jayva\Documents\GitHub\OPTIMISATION\DATA\historical flight data\AMS-BCN\CSV'

# Create a new folder for the output CSV files within the original CSV folder
output_directory_path = os.path.join(input_directory_path, 'bs')
os.makedirs(output_directory_path, exist_ok=True)

# Loop through each CSV file in the input directory
for filename in os.listdir(input_directory_path):
    if filename.endswith('.csv'):
        input_file_path = os.path.join(input_directory_path, filename)

        # Load the CSV file into a DataFrame with low_memory=False
        df = pd.read_csv(input_file_path, low_memory=False)

        # Initialize a new column for 'Flight Phases'
        df['Flight Phases'] = ''

        # Initialize a new column for 'Altitude Change'
        df['Altitude Change'] = 0  # Initialize to 0 for the first row

        # Calculate the change in altitude for each 300-second interval
        for i in range(0, len(df) - 300, 300):
            start_index = i
            end_index = min(i + 300, len(df) - 1)

            # Calculate the altitude change for the current interval
            df.loc[start_index:end_index, 'Altitude Change'] = df['[3d Altitude Ft]'].diff(periods=300)

            # Define a function to classify flight phases based on altitude change
            def classify_flight_phase(change):
                if change > 500:
                    return 'Climb'
                elif change < -700:
                    return 'Descent'
                else:
                    return 'Cruise'

            # Apply the function to create the 'Flight Phases' column for the current interval
            df.loc[end_index, 'Flight Phases'] = classify_flight_phase(df.loc[end_index, 'Altitude Change'])

        # Backfill the Flight Phases column
        df['Flight Phases'] = df['Flight Phases'].replace('', method='bfill')

        # Save the modified DataFrame to a new CSV file in the 'bs' folder with a numerical name
        output_file_path = os.path.join(output_directory_path, f'{filename.split(".")[0]}_with_phases_and_changes.csv')
        df.to_csv(output_file_path, index=False)
