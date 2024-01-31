import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from geographiclib.geodesic import Geodesic

# Load the CSV file into a pandas DataFrame excluding the last row
csv_path = r'C:\Users\jayva\Documents\GitHub\OPTIMISATION\PSO\ROUTE output\BCN-AMS.csv'
df = pd.read_csv(csv_path, engine='python')  # Skip the last row

# Convert latitude and longitude columns to numeric
df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')

# Drop rows with NaN values in Latitude or Longitude columns
df = df.dropna(subset=['Latitude', 'Longitude'])

# Extract latitude and longitude columns from the DataFrame
latitudes = df['Latitude'].tolist()
longitudes = df['Longitude'].tolist()

# Create a great circle path using geographiclib
geod = Geodesic.WGS84

# Starting and ending points
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

# Create a Cartopy GeoAxes with PlateCarree projection
fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()}, figsize=(12, 8))

# Add coastlines and countries to the map
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS, linestyle=':')

# Draw waypoints
ax.scatter(longitudes, latitudes, s=50, color='blue', label='Waypoints')

# Draw great circle path
ax.plot(gc_longitudes, gc_latitudes, linewidth=2, color='red', label='Great Circle Path')

# Set the extent of the GeoAxes explicitly
ax.set_extent([min(longitudes)-6, max(longitudes)+6, min(latitudes)-2, max(latitudes)+2])

# Show the plot
plt.title('Great Circle Path and Waypoints')
plt.legend()
plt.show()
