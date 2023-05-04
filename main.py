import yaml
import numpy as np
import h5py

config = {
    'events': [
        "/Volumes/Projects2023/ev0000364000.h5",
        "/Volumes/Projects2023/ev0000447288.h5",
        "/Volumes/Projects2023/ev0000593283.h5",
        "/Volumes/Projects2023/ev0000734973.h5",
        "/Volumes/Projects2023/ev0000773200.h5",
        "/Volumes/Projects2023/ev0001903830.h5",
        "/Volumes/Projects2023/ev0002128689.h5",
    ],
    'start': 8640000,  # 24 hours in hundredths of a second
    'end': 9000000,  # 25 hours in hundredths of a second
    'num_channels': 3,
    'output_file': 'exercise.npy',
}

with open('config.yaml', 'w') as f:
    yaml.safe_dump(config, f)
# 将YAML文本写入文件


# with open('/Users/wuyifan/Desktop/example.yaml', 'r') as file:
    # yaml_content = yaml.safe_load(file)
# print(yaml_content)


with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

for file_path in config['events']:
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
    extracted_data = np.empty((len(stations_list), config['num_channels'], config['end'] - config['start']))

    # Extract data from each station
    for i, station in enumerate(stations_list):
        extracted_data[i, 0:config['num_channels'], :] = f[station][0:config['num_channels'], config['start']:config['end']]
    print("The shape of extracted data:", extracted_data.shape)

    # Save the extracted data
    # output_file = file_path + '_' + config['output_file']
    # np.save(output_file, extracted_data)
