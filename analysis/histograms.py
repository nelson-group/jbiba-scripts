import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial import cKDTree
import matplotlib.animation as animation
from scipy.optimize import curve_fit
import time
import scipy.stats as stats


snaps = list(range(100, 1000, 50))

cmap = plt.get_cmap('Greys')

for i, snap in enumerate(snaps):
    filename= f"real-turbtest-64/snap_{snap:03d}.hdf5"
    with h5py.File(filename, 'r') as file:
        gas = file["PartType0"]
        velocities = np.array(gas["Velocities"])   
        coordinates = np.array(gas["Coordinates"])
        density = np.array(gas["Density"])

    density_function = stats.gaussian_kde(np.linalg.norm(velocities, axis=1))
    x = np.linspace(0, 1, 100)
    plt.plot(x, density_function(x), color=cmap(i/len(snaps)))

last_time = (snaps[-1] * 0.01) * 0.978462

plt.figtext(0.95, 0.95, f'Time = {last_time:.3f} Mio. yr', ha='right', va='top', fontsize=12, transform=plt.gca().transAxes)

plt.xlabel(r'$v [km/s]$') # \rho [M_{\odot}/kpc^3]
plt.title(r"Estimated PDF for $v$, Resolution $64^3$")
plt.savefig(f"velocity_histogram_64_rainbow.png")