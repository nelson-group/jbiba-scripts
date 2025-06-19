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


with h5py.File("data/turbulence30kpc.h5", "r") as f:
    table = dict()
    for key in f: 
        table[key] = np.array(f[key])

boxes = 20
smoothing = 4

table["galactic_cos"] = np.zeros(boxes**3 * len(nicegalaxy.galaxies))


for galaxy_idx in tqdm(range(len(nicegalaxy.galaxies)), desc="Looping over Galaxies"):
    galaxy = nicegalaxy.Galaxy(nicegalaxy.galaxies[galaxy_idx])
    box_length = galaxy.gas["Coordinates"][:, 0].max() - galaxy.gas["Coordinates"][:, 0].min()

    min_corner = galaxy.gas["Coordinates"].min(axis=0)
    max_corner = galaxy.gas["Coordinates"].max(axis=0)
    cutout_size = box_length/2**smoothing

    spacing = (max_corner - min_corner - cutout_size) / (boxes - 1)
    for i, j, k in tqdm(np.ndindex(boxes, boxes, boxes), total=boxes**3, desc="Processing cutouts"):
        cutout_min = min_corner + np.array([i, j, k]) * spacing
        cutout_max = cutout_min + cutout_size
        in_box = (
            (galaxy.gas["Coordinates"][:, 0] >= cutout_min[0]) & (galaxy.gas["Coordinates"][:, 0] < cutout_max[0]) &
            (galaxy.gas["Coordinates"][:, 1] >= cutout_min[1]) & (galaxy.gas["Coordinates"][:, 1] < cutout_max[1]) &
            (galaxy.gas["Coordinates"][:, 2] >= cutout_min[2]) & (galaxy.gas["Coordinates"][:, 2] < cutout_max[2])
        )
        center = 0.5 * (cutout_max + cutout_min)
        costheta = np.dot(galaxy.normal_vector, center - galaxy.bhs["Coordinates"][0])/np.linalg.norm(center - galaxy.bhs["Coordinates"][0])
        assert nicegalaxy.galaxies[galaxy_idx] == table["subhaloId"][galaxy_idx * boxes**3 + i * boxes**2 + j * boxes + k]
        table["galactic_cos"][galaxy_idx * boxes**3 + i * boxes**2 + j * boxes + k]


with h5py.File('data/cgm29kpc.h5', 'w') as f:
    for key, array in table.items():
        f.create_dataset(key, data=array)
