import joblib
import numpy as np
import os
import warnings
import csv
import math
from geographiclib.geodesic import Geodesic
from functools import lru_cache

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn") #for keeping overview whilst running the code

geod = Geodesic.WGS84

# Load the pre-trained joblib model
fuel_flow_model_path = r'CODE\FF model.joblib'
fuel_flow_model = joblib.load(fuel_flow_model_path)


class Particle:
    
    def __init__(self, num_waypoints, initial_coord, final_coord, altitude, day):
        initial_coord = self.calculate_waypoint(initial_coord, final_coord, 120.0)
        final_coord = self.calculate_waypoint(final_coord, initial_coord, 90)
    
        self.waypoints = self.generate_great_circle_waypoints(initial_coord, final_coord, num_waypoints, lateral_deviation_nm=20.0)
        self.velocity = np.random.rand(num_waypoints, 2)
        self.best_waypoints = np.copy(self.waypoints)
        self.best_fitness = float('inf')
        self.altitude = altitude
        self.day = day


    def calculate_waypoint(self, start_coord, end_coord, distance_nm):
        result = geod.Inverse(start_coord[0], start_coord[1], end_coord[0], end_coord[1])
        initial_azimuth = result['azi1']
        new_point = geod.Direct(start_coord[0], start_coord[1], initial_azimuth, distance_nm * 1852.0)
        return new_point['lat2'], new_point['lon2']


    def is_within_boundary(self, point, start, end, max_distance_nm=20.0):
        dist_start_to_point = geod.Inverse(start[0], start[1], point[0], point[1])['s12']
        dist_point_to_end = geod.Inverse(point[0], point[1], end[0], end[1])['s12']
        dist_start_to_end = geod.Inverse(start[0], start[1], end[0], end[1])['s12']

        if abs(dist_start_to_point + dist_point_to_end - dist_start_to_end) <= max_distance_nm * 1852:
            return True
        else:
            return False


    def generate_great_circle_waypoints(self, start_coord, end_coord, num_waypoints, lateral_deviation_nm=20.0):
        waypoints = []
        result = geod.Inverse(start_coord[0], start_coord[1], end_coord[0], end_coord[1])
        initial_azimuth = result['azi1']
        total_distance = result['s12']
        step_distance = total_distance / (num_waypoints - 1)

        waypoints.append(start_coord)
        for i in range(1, num_waypoints - 1):
            valid_point = False
            while not valid_point:
                distance = step_distance * i
                lateral_distance = np.random.uniform(-lateral_deviation_nm, lateral_deviation_nm) * 1852.0  # Convert NM to meters
                azimuth_variation = np.random.uniform(-30, 30)  # Degrees
                potential_point = geod.Direct(start_coord[0], start_coord[1], initial_azimuth + azimuth_variation, distance + lateral_distance)
                if self.is_within_boundary((potential_point['lat2'], potential_point['lon2']), start_coord, end_coord, max_distance_nm=lateral_deviation_nm):
                    valid_point = True
                    waypoints.append((potential_point['lat2'], potential_point['lon2']))
        waypoints.append(end_coord)
        return waypoints


def calculate_row(lat, lon):
    lat_i = round(((lat + 90) / 180) * 301) 
    lon_i = round(((lon + 180) / 360) * 601)
    return ((lat_i - 1) * 601) + lon_i - 1


base_directory = r'DATA\weather'

@lru_cache(maxsize=None)
def load_tmp_array(day, altitude):
    tmp_path = os.path.join(base_directory, f'Day_{day}_v_2', f'TMP_date_{day}_alt_{altitude}.csv')
    return np.genfromtxt(tmp_path, delimiter=',')

@lru_cache(maxsize=None)
def load_wind_array(day, altitude):
    wind_path = os.path.join(base_directory, f'Day_{day}_v_2', f'WIND_date_{day}_alt_{altitude}.csv')
    return np.genfromtxt(wind_path, delimiter=',')

@lru_cache(maxsize=None)
def load_wdir_array(day, altitude):
    wdir_path = os.path.join(base_directory, f'Day_{day}_v_2', f'WDIR_date_{day}_alt_{altitude}.csv')
    return np.genfromtxt(wdir_path, delimiter=',')

def get_weather_data(day, altitude, latitude, longitude):
    try:
        tmp = load_tmp_array(day, altitude)
        wind = load_wind_array(day, altitude)
        wdir = load_wdir_array(day, altitude)

        row = calculate_row(latitude, longitude)
        return tmp[row][2], wind[row][2], wdir[row][2]

    except FileNotFoundError:
        print(f"File not found for Day {day} at Altitude {altitude}. Skipping...")
        return None, None, None
    

def calculate_bearing(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dLon))

    initial_bearing = math.atan2(x, y)

    # Convert bearing from radians to degrees and normalize to 0-360
    initial_bearing = math.degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360

    return bearing


def calculate_fitness(waypoints, altitude, day, speed_meters_per_second):
    total_fuel_used = 0

    for i in range(len(waypoints) - 1):
        lat, lon = waypoints[i]
        next_lat, next_lon = waypoints[i + 1]

        # Calculate distance between waypoints
        distance_leg = geod.Inverse(lat, lon, next_lat, next_lon)['s12']

        # Get weather data for the next waypoint
        temperature, wind_speed, wind_direction = get_weather_data(day, altitude, next_lat, next_lon)

        # Calculate bearing and adjust speed based on wind direction
        travel_direction = calculate_bearing(lat, lon, next_lat, next_lon)
        angle_difference = abs(wind_direction - travel_direction)
        if angle_difference > 180:
            angle_difference = 360 - angle_difference

        # Adjust speed based on headwind or tailwind
        wind_effect = wind_speed * math.cos(math.radians(angle_difference))
        effective_speed = max(speed_meters_per_second - wind_effect, 0)  # Ensure speed doesn't go negative

        # Fuel flow prediction
        predicted_fuel_flow = fuel_flow_model.predict([[temperature]])[0]

        # Time calculation with effective speed and fuel used calculation
        time_leg = distance_leg / effective_speed
        fuel_used_leg = predicted_fuel_flow * (time_leg / 3600)
        total_fuel_used += fuel_used_leg

    # Final fitness is the total fuel used
    fitness = total_fuel_used
    return fitness


def calculate_time(waypoints, speed):
    time = 0
    for i in range(len(waypoints) - 1):
        lat1, lon1 = waypoints[i]
        lat2, lon2 = waypoints[i + 1]
        distance = geod.Inverse(lat1, lon1, lat2, lon2)['s12']
        time += distance / speed

    return time


def update_velocity(particle, global_best_waypoints, inertia_weight, cognitive_weight, social_weight, speed_meters_per_second):
    # Ensure all waypoints are NumPy arrays for vectorized operations
    particle_waypoints = np.array(particle.waypoints)
    particle_best_waypoints = np.array(particle.best_waypoints)
    global_best_waypoints = np.array(global_best_waypoints)

    # Calculating inertia, cognitive, and social components of the velocity
    inertia_term = inertia_weight * particle.velocity
    cognitive_term = cognitive_weight * np.random.rand(*particle.velocity.shape) * (particle_best_waypoints - particle_waypoints)
    social_term = social_weight * np.random.rand(*particle.velocity.shape) * (global_best_waypoints - particle_waypoints)

    # Updating the velocity
    particle.velocity = inertia_term + cognitive_term + social_term

    # Calculating exploration time (optional, based on your existing code structure)
    exploration_time = np.sum(np.sqrt(np.sum((particle_waypoints[1:] - particle_waypoints[:-1]) ** 2, axis=1))) / speed_meters_per_second

    return particle.velocity, exploration_time


def update_waypoints(particle, exploration_time):
    new_waypoints = particle.waypoints + particle.velocity
    return new_waypoints, exploration_time


def pso(initial_coord, final_coord, altitude, day, num_waypoints, num_particles, num_iterations, speed_meters_per_second):
    inertia_weight = 0.5
    cognitive_weight = 1.8
    social_weight = 0.8
    
    particles = [Particle(num_waypoints, initial_coord, final_coord, altitude, day) for _ in range(num_particles)]

    global_best_particle = min(particles, key=lambda p: calculate_fitness(p.waypoints, altitude, day, speed_meters_per_second))
    global_best_waypoints = np.copy(global_best_particle.waypoints)

    for iteration in range(num_iterations):
        for particle in particles:
            fitness = calculate_fitness(particle.waypoints, altitude, day, speed_meters_per_second)

            if fitness < particle.best_fitness:
                particle.best_fitness = fitness
                particle.best_waypoints = np.copy(particle.waypoints)

            if fitness < calculate_fitness(global_best_waypoints, altitude, day, speed_meters_per_second):
                global_best_waypoints = np.copy(particle.waypoints)

            # Update velocity and get exploration time
            particle.velocity, exploration_time = update_velocity(particle, global_best_waypoints, inertia_weight, cognitive_weight, social_weight, speed_meters_per_second)

            # Update particle position and incorporate exploration time into fitness
            particle.waypoints, exploration_time = update_waypoints(particle, exploration_time)
            fitness += exploration_time

       
        print(f"Iteration {iteration + 1}, Best Fitness: {calculate_fitness(global_best_waypoints, altitude, day, speed_meters_per_second)}")

    return global_best_waypoints, calculate_fitness(global_best_waypoints, altitude, day, speed_meters_per_second)


def calculate_and_save_output(csv_path, waypoints, altitude, day, speed_meters_per_second):
    # Open CSV file for writing
    with open(csv_path, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)

        # Write header row
        csvwriter.writerow(["Latitude", "Longitude", "Burned Fuel (kg)", "Time Taken (seconds)", "Temperature Used (Celsius)", 'Estimated FF (kg/hr)'])

        total_time_taken = 0
        total_fuel_burned = 0

        for i in range(len(waypoints) - 1):
            lat, lon = waypoints[i]
            next_lat, next_lon = waypoints[i + 1]

        # Calculate distance between waypoints
            distance_leg = geod.Inverse(lat, lon, next_lat, next_lon)['s12']

        # Get weather data for the next waypoint
            temperature, wind_speed, wind_direction = get_weather_data(day, altitude, next_lat, next_lon)
            total_air_temperature = temperature - 273.15

        # Calculate bearing and adjust speed based on wind direction
            travel_direction = calculate_bearing(lat, lon, next_lat, next_lon)
            angle_difference = abs(wind_direction - travel_direction)
            if angle_difference > 180:
                angle_difference = 360 - angle_difference

        # Adjust speed based on headwind or tailwind
            wind_effect = wind_speed * math.cos(math.radians(angle_difference))
            effective_speed = max(speed_meters_per_second - wind_effect, 0)  # Ensure speed doesn't go negative

        # Fuel flow prediction
            predicted_fuel_flow = fuel_flow_model.predict([[total_air_temperature]])[0]

        # Time calculation with effective speed and fuel used calculation
            time_leg = (distance_leg / effective_speed) / 3600
            fuel_used_leg = predicted_fuel_flow * time_leg
            total_fuel_burned += fuel_used_leg

            # Calculate exploration time for the leg
            total_time_taken += time_leg

            # Write row to CSV
            csvwriter.writerow([lat, lon, fuel_used_leg, time_leg, total_air_temperature, predicted_fuel_flow])

            # Debugging line to print temperature values for each leg
            print(f"Leg {i+1}: Latitude {lat}, Longitude {lon}, Temperature Used: {total_air_temperature} Celsius")

        # Write total values to the CSV file without using the word "Total"
        csvwriter.writerow(["", "", total_fuel_burned, total_time_taken, "", ""])


# Example usage
initial_coord = np.array([41.295277, 2.090804]) 
final_coord = np.array([52.302011, 4.781655])
altitude = 1  # for weather data
day = 1  # for weather data
num_waypoints = 15
num_particles = 1000
num_iterations = 60
speed_meters_per_second = 227.435889  # determined by historic data

best_waypoints, best_fitness = pso(initial_coord, final_coord, altitude, day, num_waypoints, num_particles, num_iterations, speed_meters_per_second)

# Calculate time based on the best waypoints and speed
time_taken = calculate_time(best_waypoints, speed_meters_per_second)

# Save waypoints as a CSV file with burned fuel and time information
csv_directory = r'ROUTE output'
csv_path = os.path.join(csv_directory, "BCN-AMS.csv")

calculate_and_save_output(csv_path, best_waypoints, altitude, day, speed_meters_per_second)

print(f"Best waypoints with fuel and time info saved to {csv_path}")
