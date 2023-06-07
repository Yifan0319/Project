import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft,ifft
from scipy import signal

#Initialization
file_path = r"data\ev0000364000.h5"

# Open the file
f = h5py.File(file_path, 'r')

# Extract a portion of the data from numerous stations

# List of stations to consider
stations_list = list(f.keys())

# Start and end index values for the portion of data to extract
# 24*3600*100 is 24 hours of data at 100 Hzone day of data.
# this example outputs firt hour of second day:
hour_interval = 3600*100
calc_num = 600*5
cnt = 750
start = 24*3600*100
end = 25*3600*100

# Number of channels
num_channels = 3

#extract every 5min data in each hour
data = np.empty((cnt,num_channels,calc_num,))
for i in range(cnt):
    data[i,0:num_channels,:] = f[stations_list[0]][0:num_channels,0+hour_interval*i:calc_num+hour_interval*i]

#convert the extracted data with stft method   
data_processed = np.empty((251,16,cnt,))
for i in range(cnt):
    freq, t, nd = signal.stft(data[i,1,:] ,fs = 200,window ='hann',nperseg = 500,noverlap = 300)
    data_processed[:,:,i] = nd

#save the processed data	
np.save('data\data.npy', data)
np.save('data\data_processed.npy', data_processed)