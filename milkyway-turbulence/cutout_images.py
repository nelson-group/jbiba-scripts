import numpy as np
import matplotlib.pyplot as plt
import nicegalaxy
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import binned_statistic


galaxy = nicegalaxy.Galaxy(342447)

box_length = galaxy.gas["Coordinates"][:, 0].max() - galaxy.gas["Coordinates"][:, 0].min()

min_corner = galaxy.gas["Coordinates"].min(axis=0)
max_corner = galaxy.gas["Coordinates"].max(axis=0)
N = 10
cutout_size = box_length/2**3
spacing = (max_corner - min_corner - cutout_size) / (N - 1)

result = np.zeros((N, N, N, 6))

distances = np.linalg.norm(galaxy.gas["Coordinates"]- galaxy.bhs["Coordinates"][0], axis=-1)
bin_edges = np.logspace(np.log10(distances.min()), np.log10(distances.max()), 500)
vols = galaxy.gas["Masses"]/galaxy.gas["Density"]
hist1, _, _ = binned_statistic(distances, galaxy.gas["Density"] * vols, bins=bin_edges, statistic='sum')
hist2, _, _ = binned_statistic(distances, vols, bins=bin_edges, statistic='sum')
with np.errstate(divide='ignore', invalid='ignore'):
    hist = np.where((~np.isnan(hist2)) & (~np.isnan(hist1)) & (hist2 != 0), hist1 / hist2, 0.0)
dist_bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

zero_indices = np.where(hist == 0)[0]
nonzero_indices = np.where(hist != 0)[0]
for i in zero_indices:
    closest_j = nonzero_indices[np.argmin(np.abs(dist_bin_centers[nonzero_indices] - dist_bin_centers[i]))]
    hist[i] = hist[closest_j]
distances = np.linalg.norm(galaxy.gas["Coordinates"]- galaxy.bhs["Coordinates"][0], axis=-1)
idx = np.abs(distances[:, None] - dist_bin_centers[~np.isnan(hist)& (hist != 0)][None, :]).argmin(axis=1)
dens_copy = hist[idx]

# for i, j, k in tqdm(np.ndindex(N, N, N), total=N**3, desc="Processing cutouts"):
for i in tqdm(range(7), desc="Processing cutouts"):
    cutout_min = min_corner + np.array([i, i, i]) * spacing
    cutout_max = cutout_min + cutout_size
    in_box = (
        (galaxy.gas["Coordinates"][:, 0] >= cutout_min[0]) & (galaxy.gas["Coordinates"][:, 0] < cutout_max[0]) &
        (galaxy.gas["Coordinates"][:, 1] >= cutout_min[1]) & (galaxy.gas["Coordinates"][:, 1] < cutout_max[1]) &
        (galaxy.gas["Coordinates"][:, 2] >= cutout_min[2]) & (galaxy.gas["Coordinates"][:, 2] < cutout_max[2])
    )
    fig, ax = plt.subplots()
    grid = (range(500), range(500), range(250, 251))
    structured_gas = nicegalaxy.structured_column(galaxy.gas["Coordinates"][in_box], np.log(galaxy.gas["Density"][in_box]/dens_copy[in_box]), grid_size=500)
    scalebar_length = round(cutout_size/10)
    scalebar_pixels = scalebar_length/cutout_size * structured_gas.shape[0]
    center = 0.5 * (cutout_max + cutout_min)
    closest_bh_dist = np.min(np.linalg.norm(galaxy.bhs["Coordinates"] - center, axis=-1))
    gc_dist = np.linalg.norm(np.linalg.norm(galaxy.bhs["Coordinates"][0] - center))
    im = ax.imshow(structured_gas, cmap="Blues_r", interpolation=None)
    ax.plot([structured_gas.shape[0]*(1 - 1/40) - scalebar_pixels, structured_gas.shape[0]*(1 - 1/40)], [structured_gas.shape[0]/40, structured_gas.shape[0]/40], color='white', linewidth=1)
    ax.text(structured_gas.shape[0]*(1 - 1/40) - scalebar_pixels/2, structured_gas.shape[0]/30, f"{scalebar_length} kpc", ha="center", va="top", color="white")
    ax.text(structured_gas.shape[0]*1/40, structured_gas.shape[0]*1/40, rf"$d_{{\mathrm{{BH}}}} = {closest_bh_dist:.1f} \mathrm{{kpc}}, d_{{\mathrm{{GC}}}} = {gc_dist:.1f} \mathrm{{kpc}}$", va="top", color="white")
    ax.set_xticks([])
    ax.set_yticks([])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, label=r"$\mathrm{log}(\rho_{\mathrm{col}}) [M_{\odot}/\mathrm{kpc}^3]$")
    fig.savefig(f"cutout-images/dens-contrast/cutout_{i}_{i}_{i}_{gc_dist:.1f}.png", c, dpi=500)
    plt.close(fig)