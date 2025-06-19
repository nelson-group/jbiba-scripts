import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial import cKDTree
import matplotlib.animation as animation
import sys  


sims = ["driving128-nonsol", "driving64"]
sim = sims[1]
 
vals = list()
for i in range(0, 600):
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
    v_rms = np.sqrt(np.mean(np.sum(velocities**2, axis=1)))
    a_rms = np.sqrt(np.mean(np.sum(acc**2, axis=1)))
    vals.append([c_s, v_rms, v_rms/c_s, a_rms])

# print(vals)

vals = np.array(vals)

#plt.plot(vals[:, 0], label='c_s')
plt.plot(vals[:, 1], label='v_rms')
plt.plot(vals[:, 2], label='mach')
#plt.plot(vals[:, 3], label='a_rms')
# plt.yscale("log")662
plt.legend()
plt.savefig(f"{sim}/mach.png")