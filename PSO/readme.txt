All scripts should have set relative directories so it all should work after downloading

PSO.py is the main algorithm. and the main folder also has the linear regression code that was used. 

map SHADY contains the script I made for making the standard route for the exhaustive search algorith.

map ROUTE output contains all the routes the pso algorithm made.
each csv files contains burned fuel per leg and time to fly the leg 

map DATA > para.py shows the fuel flow against temperature and speed.
in this map the folder for weather data is located
also historical flight data is stored here. in this map it shows all scripts that were used to filter the data.
it was ran in this order
1. deleting no input.py 
2. assigning phases.py (makes a new folder)
3. filter cruise.py (saves in cleaned data folder)
4. condition.py (deletes unwanted data) 

map CODE
contains plotting waypoints.py you can plot all routes that are generated in the map shady and route output
also contains weatther.py that get the weather data from closest weather datapoint
