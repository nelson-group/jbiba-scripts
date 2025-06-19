import glob, os
import numpy as np
import matplotlib.pyplot as plt
import nicegalaxy
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import h5py
from matplotlib.colors import LinearSegmentedColormap
from numba import njit, prange
from scipy.spatial import cKDTree
from matplotlib.ticker import ScalarFormatter


def get_velocity_filenames():
    cutout_path = "/u/jbiba/projects/milkyway-turbulence/data/turb-magnetic"
    galaxies_hdf5 = glob.glob(os.path.join(cutout_path, '*.h5'))
    galaxies = [int((os.path.basename(f)).split(".")[0]) for f in galaxies_hdf5]
    return galaxies


def get_turb_velocities(id):
    turb = dict()
    with h5py.File("/u/jbiba/projects/milkyway-turbulence/data/turb-magnetic/" + f"{id}.h5", "r") as f:
        for key in f:
            turb[key] = np.array(f[key])
    return turb


turb_galaxies = get_velocity_filenames()

for i in range(len(turb_galaxies)):
    turb = get_turb_velocities(turb_galaxies[i])
    galaxy = nicegalaxy.Galaxy(turb_galaxies[i])

    structured_vels = nicegalaxy.structured_column(galaxy.gas["Coordinates"], np.sqrt(np.sum(galaxy.gas["MagneticField"]**2, axis=-1)), grid_size=500)
    structured_vels_back = nicegalaxy.structured_column(galaxy.gas["Coordinates"], np.sqrt(np.sum((galaxy.gas["MagneticField"] - turb["TurbMagneticField"])**2, axis=-1)), grid_size=500)
    structured_vels_turb = nicegalaxy.structured_column(galaxy.gas["Coordinates"], np.sqrt(np.sum(turb["TurbMagneticField"]**2, axis=-1)), grid_size=500)
    structured_scale = nicegalaxy.structured_column(galaxy.gas["Coordinates"], turb["CoherenceScale"], grid_size=500)

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))

    cutout_size = galaxy.gas["Coordinates"][:, 0].max() - galaxy.gas["Coordinates"][:, 0].min()
    scalebar_length = round(cutout_size/10)
    scalebar_pixels = scalebar_length/cutout_size * structured_vels.shape[0]

    vmin = structured_vels.min()
    vmax = structured_vels.max()

    im00 = axs[0, 0].imshow(structured_vels, interpolation=None, cmap="cividis_r", norm="log")
    im01 = axs[0, 1].imshow(structured_vels_back, interpolation=None, cmap="cividis_r", norm="log")
    im10 = axs[1, 0].imshow(structured_vels_turb, interpolation=None, cmap="cividis_r", norm="log")
    im11 = axs[1, 1].imshow(structured_scale, interpolation=None, cmap="viridis", norm="log")
    ims = [[im00, im01], [im10, im11]]

    labels = [[r"$|v|$ [km/s]", r"$|v|$ [km/s]"], [r"$|v|$ [km/s]", r"$d_\mathrm{smooth}$ [kpc]"]]

    for k in range(2):
        for j in range(2):
            axs[k, j].set_xticks([])
            axs[k, j].set_yticks([])
            axs[k, j].plot([structured_vels.shape[0]*(1 - 1/40) - scalebar_pixels, structured_vels.shape[0]*(1 - 1/40)], [structured_vels.shape[0]/40, structured_vels.shape[0]/40], color='white', linewidth=1)
            axs[k, j].text(structured_vels.shape[0]*(1 - 1/40) - scalebar_pixels/2, structured_vels.shape[0]/30, f"{scalebar_length} kpc", ha="center", va="top", color="white")
            divider = make_axes_locatable(axs[k, j])
            cax = divider.append_axes("bottom", size="5%", pad=0.05)
            cbar = plt.colorbar(ims[k][j], cax=cax, orientation="horizontal", label=labels[k][j])

    fig.tight_layout()
    fig.savefig(f"turb-magnetic-images/{turb_galaxies[i]}-panels.png", dpi=500)