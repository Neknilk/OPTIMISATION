import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geographiclib.geodesic import Geodesic
import os

# Directory containing the CSV files
directory = r'C:\Users\jayva\Documents\GitHub\OPTIMISATION\PSO\ROUTE output'

# List all CSV files in the directory
csv_files = [file for file in os.listdir(directory) if file.endswith('.csv')]

# Create a Cartopy GeoAxes with PlateCarree projection
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8))

# Add coastlines and countries to the map
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Initialize geodesic
geod = Geodesic.WGS84

# Process each CSV file
for csv_file in csv_files:
    csv_path = os.path.join(directory, csv_file)
    df = pd.read_csv(csv_path, engine='python')

    # Convert latitude and longitude columns to numeric and drop NaNs
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df = df.dropna(subset=['Latitude', 'Longitude'])

    # Extract latitude and longitude
    latitudes = df['Latitude'].tolist()
    longitudes = df['Longitude'].tolist()

    # Starting and ending points for great circle path
    start = (latitudes[0], longitudes[0])
    end = (latitudes[-1], longitudes[-1])

    # Calculate geodesic path
    g = geod.Inverse(start[0], start[1], end[0], end[1])
    l = geod.Line(g['lat1'], g['lon1'], g['azi1'])
    num_points = 100

    gc_latitudes, gc_longitudes = [], []

    for i in range(num_points + 1):
        pos = l.Position(i * g['s12'] / num_points)
        gc_latitudes.append(pos['lat2'])
        gc_longitudes.append(pos['lon2'])

    # Plot waypoints and great circle path for each CSV file
    ax.scatter(longitudes, latitudes, s=50, label=f'Waypoints {csv_file}')
    ax.plot(gc_longitudes, gc_latitudes, linewidth=2, label=f'Great Circle {csv_file}')

# Set the extent of the GeoAxes explicitly (adjust these values as needed)
# ax.set_extent([min_longitude-6, max_longitude+6, min_latitude-2, max_latitude+2])

# Show the plot
plt.title('Great Circle Paths and Waypoints for All Routes')
plt.show()
