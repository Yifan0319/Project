import yaml
import numpy as np
import h5py

with open('/Users/wuyifan/Desktop/example.yaml', 'r') as f:
    config = yaml.safe_load(f)

event_paths = config['data']['load']['events']
n_stations = config['filter']['n_stations']

for file_path in event_paths:
    # Open the file
    f = h5py.File(file_path, 'r')
    stations_list = list(f.keys())

    # Get the distances for all stations
    distances = []
    for k in f.keys():
        stations_distance = np.array(f[k].attrs['dist_m'].split(','), dtype=float)
        distances.extend(stations_distance)

    # Sort the distances and select the closest n_stations
    distances = np.array(distances)
    closest_indices = np.argsort(distances)[:n_stations]
    closest_stations = [stations_list[i] for i in closest_indices]

    print("Closest stations:")
    for station in closest_stations:
        print(station)

    # Allocate space for the extracted data
    extracted_data = np.empty((len(stations_list), n_stations, config['filter']['end'] - config['filter']['start']))

    # Extract data from each station
    for i, station in enumerate(stations_list):
        extracted_data[i, :, :] = f[station][closest_indices, config['filter']['start']:config['filter']['end']]
    print("The shape of extracted data:", extracted_data.shape)

    # Save the extracted data
    # output_file = file_path + '_' + config['data']['save']['filename']
    # np.save(output_file, extracted_data)
