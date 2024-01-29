import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing CSV files
directory = r'PSO\DATA\historical flight data\AMS-BCN\CSV\cleaned'

# Get a list of CSV files in the directory
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# Create a color map for different flight phases
phase_colors = {'climb': 'blue', 'descent': 'green', 'cruise': 'purple'}

# Create a 2D plot
fig, ax = plt.subplots(figsize=(12, 8))

# Iterate through each CSV file and plot time (x) vs altitude (y) with different colors for each flight phase
for i, file in enumerate(csv_files):
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path, low_memory=False)

    # Drop rows with NaN values in 'Time (secs)' or '[3d Altitude Ft]'
    df = df.dropna(subset=['Time (secs)', '[3d Altitude Ft]'])

    # Filter data for cruise phase and altitude above 20000
    cruise_data = df[(df['Flight Phases'].str.lower() == 'cruise') & (df['[3d Altitude Ft]'] > 20000)]

    # Plot climb and descent phases without altitude condition
    for phase, color in phase_colors.items():
        if phase.lower() != 'cruise':
            phase_data = df[df['Flight Phases'].str.lower() == phase]
            ax.plot(phase_data['Time (secs)'], phase_data['[3d Altitude Ft]'],
                    label=f'{file} - {phase}', color=color)

    if not cruise_data.empty:
        # Plot cruise data with a specific color and altitude condition
        ax.plot(cruise_data['Time (secs)'], cruise_data['[3d Altitude Ft]'],
                label=f'{file} - Cruise', color=phase_colors['cruise'])

# Set labels, legend, and axis limits
ax.set_title('Flight Paths with Altitude > 20000')
ax.set_xlabel('Time (secs)')
ax.set_ylabel('[3d Altitude Ft]')
ax.legend()
ax.set_xlim(0, 8000)  # Set x-axis limits
ax.set_ylim(-10, 45000)  # Set y-axis limits

# Show the plot
plt.grid(True)
plt.show()