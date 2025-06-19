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


def get_magnetic_filenames():
    cutout_path = "/u/jbiba/projects/milkyway-turbulence/data/turb-magnetic"
    galaxies_hdf5 = glob.glob(os.path.join(cutout_path, '*.h5'))
    galaxies = [int((os.path.basename(f)).split(".")[0]) for f in galaxies_hdf5]
    return galaxies


def get_turb_mfields(id):
    turb = dict()
    with h5py.File("/u/jbiba/projects/milkyway-turbulence/data/turb-magnetic/" + f"{id}.h5", "r") as f:
        for key in f:
            turb[key] = np.array(f[key])
    return turb


turb_galaxies = get_magnetic_filenames()
# turb_galaxies = [508538]

for idx in turb_galaxies:
    turb = get_turb_mfields(idx)
    galaxy = nicegalaxy.Galaxy(idx)

    basis_change = (galaxy.gas["Coordinates"][1000:1003] - galaxy.bhs["Coordinates"][0])
    transform = galaxy.gas["RotatedCoordinates"][1000:1003]
    rotation = (np.linalg.inv(basis_change) @ transform).T


    new_rotated = (galaxy.gas["Coordinates"] - galaxy.bhs["Coordinates"][0]) @ rotation.T
    rotated_magneticfield_turb = turb["TurbMagneticField"] @ rotation.T
    rotated_magneticfield_bulk = (galaxy.gas["MagneticField"] - turb["TurbMagneticField"]) @ rotation.T

    mask = np.linalg.norm(new_rotated, ord=np.inf, axis=-1) < 50
    coords = new_rotated[mask]
    bulk_magnetic = rotated_magneticfield_bulk[mask] 
    turb_magnetic = rotated_magneticfield_turb[mask]

    # coords[:, [0, 2]] = coords[:, [2, 0]]
    # bulk_magnetic[:, [0, 2]] = bulk_magnetic[:, [2, 0]]
    # turb_magnetic[:, [0, 2]] = turb_magnetic[:, [2, 0]]
    grid_size = 500
    grid = (range(0, grid_size), range(0, grid_size), range(grid_size//2, grid_size//2 + 1))
    struct_bulk = nicegalaxy.map_unstructured_to_structured_slice_optimized(coords, bulk_magnetic, grid_size=grid_size, grid=grid)
    Y, X = np.mgrid[0:grid_size, 0:grid_size]
    U_bulk = struct_bulk[:, :, 0]
    V_bulk = struct_bulk[:, :, 1]

    struct_turb = nicegalaxy.map_unstructured_to_structured_slice_optimized(coords, turb_magnetic, grid_size=grid_size, grid=grid)
    Y, X = np.mgrid[0:grid_size, 0:grid_size]
    U_turb = struct_turb[:, :, 0]
    V_turb = struct_turb[:, :, 1]

    struct_dens = nicegalaxy.map_unstructured_to_structured_slice_optimized(coords, galaxy.gas["Density"][mask],grid_size=grid_size, grid=grid)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    cutout_size = 100
    scalebar_length = round(cutout_size/10)
    scalebar_pixels = scalebar_length/cutout_size * struct_dens.shape[0]
    ax[0].imshow(struct_dens, cmap="Blues_r", norm="log", interpolation=None, origin="lower")
    ax[0].set_yticks([])
    ax[0].set_xticks([])
    ax[0].streamplot(X, Y, U_bulk, V_bulk, density=3, color=(0, 0, 0, 0.3), arrowsize=0, linewidth=0.5, broken_streamlines=False)
    ax[1].imshow(struct_dens, cmap="Blues_r", norm="log", interpolation=None, origin="lower")
    ax[1].set_yticks([])
    ax[1].set_xticks([])
    ax[1].streamplot(X, Y, U_turb, V_turb, density=3, color=(0, 0, 0, 0.3), arrowsize=0, linewidth=0.5, broken_streamlines=False)
    ax.plot([struct_dens.shape[0]*(1 - 1/40) - scalebar_pixels, struct_dens.shape[0]*(1 - 1/40)], [struct_dens.shape[0]*(1 - 1/40), struct_dens.shape[0]*(1 - 1/40)], color='white', linewidth=1)
    ax.text(struct_dens.shape[0]*(1 - 1/40) - scalebar_pixels/2, struct_dens.shape[0]*(1 - 1/30), f"{scalebar_length} kpc", ha="center", va="top", color="white")
    fig.tight_layout()
    fig.savefig(f"turb-velocity-images/disk/{idx}.png", bbox_inches="tight", dpi=500)
