import yaml
import numpy as np
import h5py
import os


with open('/Users/wuyifan/Desktop/example.yaml', 'r') as file:
    config = yaml.safe_load(file)

event_paths = config['data']['load']['events']
n_stations = config['filter']['n_stations']
repository_path = config['data']['load']['repository']

if event_paths == "ALL":
    # Get all file names in the repository directory
    event_files = [os.path.join(repository_path, file) for file in os.listdir(repository_path)]
else:
    event_files = event_paths  # Use the provided file paths

for file_path in event_files:
    # Open the file
    f = h5py.File(file_path, 'r')
    stations_list = list(f.keys())

    # Get the closest station to the event
    min_distance = float("inf")
    min_station = ""
    for k in f.keys():
        stations_distance = np.array(f[k].attrs['dist_m'].split(','), dtype=float)
        closest_distance = np.min(stations_distance)
        print(f"{k}: {closest_distance}")
        if closest_distance < min_distance:
            min_distance = closest_distance
            min_station = k
    print(f"The closest distance is {min_distance}, found in station {min_station}")

    # Allocate space for the extracted data
    extracted_data = np.empty((len(stations_list), n_stations, config['filter']['end'] - config['filter']['start']))

    # Extract data from each station
    for i, station in enumerate(stations_list):
        extracted_data[i, 0:n_stations, :] = f[station][0:n_stations, config['filter']['start']:config['filter']['end']]
    print("The shape of extracted data:", extracted_data.shape)

    # Save the extracted data
    # output_file = file_path + '_' + config['data']['save']['filename']
    # np.save(output_file, extracted_data)
