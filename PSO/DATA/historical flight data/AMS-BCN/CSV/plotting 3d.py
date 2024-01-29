import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Directory containing CSV files
directory = r'C:\Users\jayva\Documents\GitHub\OPTIMISATION\DATA\historical flight data\AMS-BCN\CSV\cleaned'

# Get a list of CSV files in the directory
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Iterate through each CSV file and plot latitude, longitude, and altitude
for file in csv_files:
    file_path = os.path.join(directory, file)
    df = pd.read_csv(file_path, low_memory=False)
    latitude = df['[3d Latitude]']
    longitude = df['[3d Longitude]']
    altitude = df['[3d Altitude Ft]']
    ax.scatter(longitude, latitude, altitude, s=10, alpha=0.5, label=file)

# Set the range for both x-axis (longitude) and y-axis (latitude)
ax.set_xlim(-180, 180)
ax.set_ylim(10, 90)

# Reverse the y-axis for latitude
ax.invert_yaxis()

# Set the range for the z-axis (altitude)
ax.set_zlim(-10, 40000)

# Set labels and legend
ax.set_title('Flight Paths Overlay with Altitude')
ax.set_xlabel('[3d Longitude]')
ax.set_ylabel('[3d Latitude]')
ax.set_zlabel('[3d Altitude Ft]')
ax.legend()

# Show the plot
plt.grid(True)
plt.show()
