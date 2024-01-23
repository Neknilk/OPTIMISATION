import math
import csv

# Constants for nautical miles conversion and exclusion zones
KM_TO_NAUTICAL_MILES = 0.539957
EXCLUSION_START_NM = 120  # Exclusion zone at the start in nautical miles
EXCLUSION_END_NM = 90  # Exclusion zone at the end in nautical miles

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0  # Radius of Earth in kilometers
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c  # Distance in kilometers

def calculate_initial_bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    initial_bearing = math.atan2(x, y)
    return (math.degrees(initial_bearing) + 360) % 360  # Bearing in degrees

def calculate_waypoint(lat1, lon1, bearing, distance_km):
    R = 6371.0  # Radius of Earth in kilometers
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    bearing = math.radians(bearing)
    lat2 = math.asin(math.sin(lat1) * math.cos(distance_km / R) +
                     math.cos(lat1) * math.sin(distance_km / R) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance_km / R) * math.cos(lat1),
                             math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def generate_waypoints(lat1, lon1, lat2, lon2, num_waypoints):
    total_distance_km = haversine(lat1, lon1, lat2, lon2)
    waypoints = []
    step = total_distance_km / (num_waypoints + 1)  # +1 to distribute waypoints evenly
    bearing = calculate_initial_bearing(lat1, lon1, lat2, lon2)
    for i in range(1, num_waypoints + 1):
        distance_km = step * i
        waypoint = calculate_waypoint(lat1, lon1, bearing, distance_km)
        waypoints.append(waypoint)
    return waypoints

def main():
    start_lat, start_lon = 52.33193666208914, 4.91636699999146  # Replace with the actual start coordinates
    end_lat, end_lon = -81.30421168617204, 102.099288545431156  # Replace with the actual end coordinates
    num_waypoints = 15  # Replace with the desired number of waypoints

    # Calculate the total distance and convert it to nautical miles
    total_distance_nm = haversine(start_lat, start_lon, end_lat, end_lon) * KM_TO_NAUTICAL_MILES

    # Calculate the exclusion zones in kilometers
    exclusion_start_km = EXCLUSION_START_NM / KM_TO_NAUTICAL_MILES
    exclusion_end_km = EXCLUSION_END_NM / KM_TO_NAUTICAL_MILES
    exclusion_total_km = exclusion_start_km + exclusion_end_km

    # Adjust the total number of waypoints to account for the exclusion zones
    total_distance_without_exclusions_km = (total_distance_nm - EXCLUSION_START_NM - EXCLUSION_END_NM) / KM_TO_NAUTICAL_MILES
    waypoints = generate_waypoints(start_lat, start_lon, end_lat, end_lon, num_waypoints)

    # Filter out waypoints within the exclusion zones
    filtered_waypoints = []
    for waypoint in waypoints:
        if haversine(start_lat, start_lon, *waypoint) > exclusion_start_km and \
           haversine(*waypoint, end_lat, end_lon) > exclusion_end_km:
            filtered_waypoints.append(waypoint)

    # Write waypoints to CSV file
    csv_filename = 'C:\\Users\\jayva\\Documents\\GitHub\\OPTIMISATION\\shady\\waypoints.csv'
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['Latitude', 'Longitude']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for waypoint in filtered_waypoints:
            writer.writerow({'Latitude': waypoint[0], 'Longitude': waypoint[1]})

    print(f"Waypoints have been saved to {csv_filename}")

if __name__ == "__main__":
    main()
