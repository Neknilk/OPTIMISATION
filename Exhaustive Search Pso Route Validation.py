# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:11:07 2024

@author: Shady Habib
"""
from geopy.distance import geodesic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn") #for keeping overview whilst running the code

# Function to calculate bearing between two points
def calculate_bearing(pointA, pointB):
    lat1 = math.radians(pointA['Latitude'])
    lat2 = math.radians(pointB['Latitude'])
    diffLong = math.radians(pointB['Longitude'] - pointA['Longitude'])

    x = math.sin(diffLong) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))

    initial_bearing = math.atan2(x, y)
    initial_bearing = math.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing

# Function to offset coordinates by a given distance and bearing
def offset_coordinates(lat, lon, distance_km, bearing):
    R = 6371.01  # Earth's radius in kilometers

    bearing = math.radians(bearing)

    lat1 = math.radians(lat)
    lon1 = math.radians(lon)

    lat2 = math.asin(math.sin(lat1) * math.cos(distance_km / R) + 
                     math.cos(lat1) * math.sin(distance_km / R) * math.cos(bearing))
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance_km / R) * math.cos(lat1), 
                             math.cos(distance_km / R) - math.sin(lat1) * math.sin(lat2))

    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)

    return lat2, lon2

# Function to create parallel nodes
def create_parallel_nodes(original_node, origin_node):
    distance_between_nodes = 5  # This is the distance in kilometers for the offset
    num_parallel_nodes = 5

    bearing = calculate_bearing(origin_node, original_node)
    perpendicular_bearing = (bearing + 90) % 360  # Offset by 90 degrees to get perpendicular direction

    parallel_nodes = []

    for i in range(num_parallel_nodes):
        offset = (i - (num_parallel_nodes - 1) / 2) * distance_between_nodes
        new_node = original_node.copy()
        new_node["Latitude"], new_node["Longitude"] = offset_coordinates(
            new_node["Latitude"], new_node["Longitude"], offset, perpendicular_bearing
        )
        parallel_nodes.append(new_node)

    return parallel_nodes



# Function to calculate row index based on latitude and longitude
def calculate_row(lat, lon):
    lat_i = round((lat + 90) / 180 * 301)
    lon_i = round((lon + 180) / 360 * 601)
    return (lat_i - 1) * 601 + lon_i - 1

# Function to retrieve weather data for a given day, altitude, latitude, and longitude
def get_weather_data(day, altitude, latitude, longitude):
    base_directory = r'C:\Python Exercices\Weather Data'
    tmp_path = os.path.join(base_directory, f'TMP_date_{day}_alt_{altitude}.csv')
    wind_path = os.path.join(base_directory, f'WIND_date_{day}_alt_{altitude}.csv')
    wdir_path = os.path.join(base_directory, f'WDIR_date_{day}_alt_{altitude}.csv')

    try:
        tmp = np.genfromtxt(tmp_path, delimiter=',')
        wind = np.genfromtxt(wind_path, delimiter=',')
        wdir = np.genfromtxt(wdir_path, delimiter=',')
        row = calculate_row(latitude, longitude)
        return tmp[row][2], wind[row][2], wdir[row][2]
    except FileNotFoundError:
        print(f"File not found: {tmp_path} or {wind_path} or {wdir_path}. Skipping...")
        return None, None, None

# Function to add weather data to a single node
def add_weather_data_to_node(node, day, altitude):
    temperature, wind_speed, wind_direction = get_weather_data(day, altitude, node['Latitude'], node['Longitude'])
    node['Temperature'] = temperature
    node['Wind_Speed'] = wind_speed
    node['Wind_Direction'] = wind_direction
    return node

# Load the trained model from file
model_filename = r"C:\Python Exercices\FF model.joblib"  # Make sure to use a raw string (prefix with 'r') for file paths
model = joblib.load(model_filename)

# Function to predict fuel flow based on temperature in Celsius
def predict_fuel_flow(temperature_celsius):
    input_data = [[temperature_celsius]]
    fuel_flow_prediction = model.predict(input_data)
    return fuel_flow_prediction[0]

# Function to convert Kelvin to Celsius
def kelvin_to_celsius(temp_kelvin):
    return temp_kelvin - 273.15

# Load waypoints data
csv_path = "C:\Python Exercices\AMS-BCNv3.csv"
df = pd.read_csv(csv_path)

# Create parallel nodes and add weather data
all_nodes_with_parallel = []
day, altitude = 1, 1  # Example day and altitude, adjust as needed

# Initialize the node number counter
node_number = 0

# Add weather data to the first row (origin airport)
origin_node = add_weather_data_to_node(df.iloc[0].copy(), day, altitude)
origin_node['node_number'] = node_number  # Assign node number 0 to origin
all_nodes_with_parallel.append(origin_node)

# Increment node number for the next set of nodes
node_number += 1

for i in range(1, len(df) - 1):
    original_node = df.iloc[i]
    parallel_nodes = create_parallel_nodes(original_node, origin_node)
    for node in parallel_nodes:
        node_with_weather = add_weather_data_to_node(node, day, altitude)
        node_with_weather['node_number'] = node_number
        all_nodes_with_parallel.append(node_with_weather)
        node_number += 1  # Increment the node number for each parallel node

# Add weather data to the last row (destination airport)
last_node = add_weather_data_to_node(df.iloc[-1].copy(), day, altitude)
last_node['node_number'] = node_number  # Assign the next node number to the destination
all_nodes_with_parallel.append(last_node)

# Convert to DataFrame without sorting
graph_df = pd.DataFrame(all_nodes_with_parallel).reset_index(drop=True)


# Remove the 'node' column if present
if 'node' in graph_df.columns:
    graph_df.drop('node', axis=1, inplace=True)

# Add fuel flow predictions to the DataFrame
graph_df['Fuel_Flow'] = graph_df['Temperature'].apply(kelvin_to_celsius).apply(predict_fuel_flow)

# Visualization
plt.scatter(graph_df['Longitude'], graph_df['Latitude'], c=graph_df['node_number'], cmap='viridis', s=10)
for i, txt in enumerate(graph_df['node_number']):
    # Convert to integer before converting to string to remove the trailing .0
    plt.annotate(str(int(txt)), (graph_df['Longitude'][i], graph_df['Latitude'][i]), textcoords="offset points", xytext=(0, 5), ha='center')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Node Numbers Visualization')
plt.show()

pd.set_option('display.max_rows', None)  # Shows all rows
pd.set_option('display.max_columns', None)  # Shows all columns
pd.set_option('display.width', 1000)  # Expand display width
pd.set_option('display.max_colwidth', None)  # Show the full content of each column


# Correct way to access multiple columns
selected_columns = graph_df[['Latitude', 'Longitude', 'Temperature', 'Fuel_Flow', 'Wind_Speed', 'Wind_Direction']]
print(selected_columns)



#################################################### Graph Creation #####################################################################

# Find endpoint of graph from waypoint index
max_index = max(graph_df.index)

# Create edges from start to first waypoint
points_list = [(0,1), (0,2), (0,3), (0,4), (0,5)]

# Loop over range of waypoints to create correct edges for the graph
for i in range(5, max_index - 1, 5):
    points_list.extend([(i-4,i+1), (i-4,i+2), (i-3,i+1), (i-3,i+2), (i-3,i+3), (i-2,i+2), (i-2,i+3), (i-2,i+4), (i-1,i+3), (i-1,i+4), (i-1,i+5), (i,i+4), (i,i+5)])

# Create edges from one to last waypoint to the final option
points_list.extend([(max_index - 5, max_index), (max_index - 4, max_index), (max_index - 3, max_index), (max_index - 2, max_index), (max_index - 1, max_index)])
#print(points_list)

# Set new list for distance, origin, destination and weights
origin, destination, weight = [], [], []

# Loop over edges to extract origin, destination and the weights
for i in range(len(points_list)):
    origin.append(points_list[i][0])
    destination.append(points_list[i][1])
    

graph_data = pd.DataFrame(list(zip(origin, destination)), columns=['origin', 'destination'])



def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    initial_bearing = math.atan2(x, y)
    return (math.degrees(initial_bearing) + 360) % 360  # Bearing in degrees

def calculate_fuel_for_edge(start_node, end_node, speed_meters_per_second):
    # Extract data for start and end nodes
    lat, lon = graph_df.at[start_node, 'Latitude'], graph_df.at[start_node, 'Longitude']
    next_lat, next_lon = graph_df.at[end_node, 'Latitude'], graph_df.at[end_node, 'Longitude']
    fuel_flow = graph_df.at[start_node, 'Fuel_Flow']
    wind_speed = graph_df.at[end_node, 'Wind_Speed']
    wind_direction = graph_df.at[end_node, 'Wind_Direction']

    # Calculate distance between nodes
    distance_leg = geodesic((lat, lon), (next_lat, next_lon)).meters

    # Calculate bearing and adjust speed based on wind direction
    travel_direction = calculate_bearing(lat, lon, next_lat, next_lon)
    angle_difference = abs(wind_direction - travel_direction)
    if angle_difference > 180:
        angle_difference = 360 - angle_difference

    # Adjust speed based on headwind or tailwind
    wind_effect = wind_speed * math.cos(math.radians(angle_difference))
    effective_speed = max(speed_meters_per_second - wind_effect, 0)  # Ensure speed doesn't go negative

    # Time calculation with effective speed
    time_leg = distance_leg / effective_speed

    # Fuel used calculation
    fuel_used_leg = fuel_flow * (time_leg / 3600)  # Convert time to hours for fuel calculation

    return fuel_used_leg

# Assuming a constant speed in meters per second
speed_mps = 227.435889  # determined by historic data

# Calculate fuel for each edge and add to the DataFrame
for i in range(len(points_list)):
    start, end = points_list[i]
    fuel_used = calculate_fuel_for_edge(start, end, speed_mps)
    graph_data.at[i, 'weight'] = fuel_used
    # Also add the coordinates
    graph_data.at[i, 'origin_latitude'] = graph_df.at[start, 'Latitude']
    graph_data.at[i, 'origin_longitude'] = graph_df.at[start, 'Longitude']
    graph_data.at[i, 'destination_latitude'] = graph_df.at[end, 'Latitude']
    graph_data.at[i, 'destination_longitude'] = graph_df.at[end, 'Longitude']

print(graph_data)

#############################################PSO Route fuel burn dubble check ##########################################################3

# Extracting the specific edges from the graph_data DataFrame
specific_edges = [(0, 3), (3, 8), (8, 13), (13, 18), (18, 23), (23, 28), (28, 33), (33, 38), (38, 43), (43, 48), (48, 53), (53, 58), (58, 61)]

# Initializing total weight
total_weight = 0

# Looping through the specific edges and adding up their weights
for edge in specific_edges:
    origin, destination = edge
    # Retrieve the weight for the specific edge
    edge_weight = graph_data[(graph_data['origin'] == origin) & (graph_data['destination'] == destination)]['weight']
    if not edge_weight.empty:
        total_weight += edge_weight.values[0]
    else:
        print(f"No direct edge found between node {origin} and node {destination}")

print("Fuel burned by Pso Algorithm route:", total_weight)


###################################Exhaustive Search#########################################################



 # Define breadth-first search function
def bfs(graph,node,total_levels,trajectory):
    if node > total_levels:
         print("WARNING 01: Something went extremly wrong")
         return 2   # This is an error code    
    
    if (sum(graph[node]) == 0): # This means that this is an end node.
         trajectories.append(trajectory[:])  # The [:] is there to copy the list. Otherwise python does strange things such as replace values
         return 0   # Just to return something
    
    queue = [] #Connected nodes to current node
    exploring = graph[node] # All possible conections
    for i in range (0, len(exploring)):
         if exploring[i] != 0: # Identify consecutive nodes
             queue.append(i)   # Add the indentified nodes to explore.

    for i in queue:   # See the queue connections
         trajectory_temp = trajectory[:]
         trajectory_temp.append(i)
         status = bfs(graph,i,total_levels,trajectory_temp)
         del(trajectory_temp)
    return 0    # Just to return something

 # Initiate new list for all trajctories    
trajectories = []

 # Create matrix with 0's for every location
matrix = np.zeros((max_index + 1, max_index + 1))

 # Update matrix with edges that are present as 1's
for a in range(len(graph_data)):
     i = int(graph_data.iloc[a]['origin'])
     j = int(graph_data.iloc[a]['destination'])
     matrix[i][j] = 1

 # Initiate new list and transform matrix to graph
graph = []
for i in range(len(matrix)):
     graph.append(list(matrix[i]))

#print(graph)

 # Find total levels of graph, set initial trajectory and start BFS
total_levels = len(graph)
trajectory = [0]
status = bfs(graph,trajectory[-1],total_levels,trajectory)

 # Copy trajectories to solutions
solution = trajectories[:]

 # Initiate new list for route distances
route_distances = []

 # Loop over all solutions
for j in range(len(solution)):
     # Initiate new list for route distance and store individual route
     route_distance = []
     route = solution[j]
     # Loop over individual route
     for i in range(len(route)-1):
        # Find node number of origin and destination
         origin = route[i]
         destination = route[i+1]
         # Find weight, the edge's weight in original dataframe
         fuel_burned = graph_data[(graph_data['origin'] == origin) & (graph_data['destination'] == destination)]['weight'].values[0]
         route_distance.append(fuel_burned)
     # Find total route distance and append to list
     total_route_distance = sum(route_distance)
     route_distances.append(total_route_distance)

 # Find minimum distance from exhaustive search
exhaustive_dist = min(route_distances)
print("Fuel burned at the most economical route:", exhaustive_dist)

# Find the index of the shortest route
index_of_shortest_route = route_distances.index(exhaustive_dist)

# Retrieve the shortest route using the index
shortest_route = solution[index_of_shortest_route]

# Print the coordinates of each waypoint in the shortest route
print("Coordinates of the most economical route:")
for node in shortest_route:
    latitude = graph_df.at[node, 'Latitude']
    longitude = graph_df.at[node, 'Longitude']
    print(f"Node {node}: Latitude {latitude}, Longitude {longitude}")
    
    
#############################Both routes Visualization #####################################
    
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import networkx as nx

# Create graph
G = nx.Graph()
G.add_edges_from(points_list)

# Set node positions according to their geographic location
pos = {node: (lon, lat) for node, (lon, lat) in enumerate(zip(graph_df['Longitude'], graph_df['Latitude']))}

# Create a larger figure to better fit all nodes and edges
plt.figure(figsize=(12, 8))

# Edges from the most economical route found by the exhaustive search
economical_route_edges = [(shortest_route[i], shortest_route[i + 1]) for i in range(len(shortest_route) - 1)]

# Convert edge lists to sets for set operations
specific_edges_set = set(specific_edges)
economical_route_edges_set = set(economical_route_edges)
all_edges_set = set(G.edges())

# Draw the network without labels
nx.draw_networkx_nodes(G, pos, node_size=20, node_color='blue', alpha=0.6)

# Draw regular edges
regular_edges = all_edges_set - specific_edges_set - economical_route_edges_set
nx.draw_networkx_edges(G, pos, edgelist=regular_edges, style='dotted', edge_color='grey', alpha=0.4)

# Draw specific edges in green
specific_edges_lines = nx.draw_networkx_edges(G, pos, edgelist=specific_edges_set, edge_color='green', width=2, alpha=0.6)

# Draw economical route edges in red
economical_route_edges_lines = nx.draw_networkx_edges(G, pos, edgelist=economical_route_edges_set, edge_color='red', width=2, alpha=0.8)

# Create a legend for the colored edges
green_line = mlines.Line2D([], [], color='green', marker='_', markersize=15, label='Exhaustive Search Route')
red_line = mlines.Line2D([], [], color='red', marker='_', markersize=15, label='Optimization Algorithm Route')

# Add the legend to the plot
plt.legend(handles=[green_line, red_line], loc='upper left')

# Show the plot
plt.title('Graph Visualization with Geographic Layout')
plt.tight_layout()
plt.show()   
    
    
