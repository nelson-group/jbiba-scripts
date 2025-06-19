import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial import cKDTree
import matplotlib.animation as animation
import sys  


sims = ["driving128-nonsol", "driving64", "driving32"]

sim = sims[2]

i = 0
filename = f"{sim}/output/snap_{i:03d}.hdf5" # str(sys.argv[1])
with h5py.File(filename, 'r') as file:
    gas = file["PartType0"]
    coordinates = np.array(gas["Coordinates"])
    velocities = np.array(gas["Velocities"])   
    density = np.array(gas["Density"])
    pressure = np.array(gas["Pressure"])
    acc = np.array(gas["Acceleration"])

gamma = 5/3
soundspeed = np.sqrt(gamma*pressure/density)
c_s = np.mean(soundspeed)
print(c_s)
