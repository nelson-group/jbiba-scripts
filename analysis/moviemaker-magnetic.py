import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
import glob
import os
from pathlib import Path
from multiprocessing import Pool
from tqdm import tqdm
from functools import partial
import os
from measurments import Sim
import cmasher as cm
import nicegalaxy


sim = Sim("/u/jbiba/projects/turbulent-driving/astro-turbulence/fiducial")


def generate_frame(frame, sim):
    grid_size = 1000
    grid = (range(0, grid_size), range(0, grid_size), range(grid_size//2, grid_size//2 + 1))
    coords = sim.load(frame, "Coordinates")
    dens = sim.load(frame, "Density")
    mfield = sim.load(frame, "MagneticField")
    time_bet_snaps = 0.0002855185
    time = 978.5 * frame * time_bet_snaps
    struct_bulk = nicegalaxy.map_unstructured_to_structured_slice_optimized(coords, mfield, grid_size=grid_size, grid=grid)
    Y, X = np.mgrid[0:grid_size, 0:grid_size]
    U_bulk = struct_bulk[:, :, 0]
    V_bulk = struct_bulk[:, :, 1]
    struct_dens = nicegalaxy.map_unstructured_to_structured_slice_optimized(coords, dens,grid_size=grid_size, grid=grid)
    fig, ax = plt.subplots(figsize=(10, 10))
    cutout_size = 1
    scalebar_length = cutout_size/10
    scalebar_pixels = scalebar_length/cutout_size * struct_dens.shape[0]
    im = ax.imshow(np.log10(struct_dens/np.mean(struct_dens)), cmap=cm.iceburn, interpolation=None, vmin=-0.15, vmax=0.1, origin="lower")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.streamplot(X, Y, U_bulk, V_bulk, density=3, color=(1, 1, 1, 0.3), arrowsize=0, linewidth=0.5, broken_streamlines=False)
    ax.plot([struct_dens.shape[0]*(1 - 1/40) - scalebar_pixels, struct_dens.shape[0]*(1 - 1/40)], [struct_dens.shape[0]*(1 - 1/40), struct_dens.shape[0]*(1 - 1/40)], color='white', linewidth=1)
    ax.text(struct_dens.shape[0]*(1 - 1/40) - scalebar_pixels/2, struct_dens.shape[0]*(1- 1/30), f"{round(scalebar_length*1000)} pc", ha="center", va="top", color="white")
    ax.text(struct_dens.shape[0]*(1 - 1/40) - scalebar_pixels/2, struct_dens.shape[0] * (1/30), f"t = {time:.2f} Myr", ha="center", va="top", color="white")
    # fig.tight_layout()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax=cax, label=r"$\ln(\rho/\langle\rho\rangle)$")
    fig.savefig(f"/u/jbiba/projects/turbulent-driving/astro-turbulence/fiducial/plots/movie_corr/{frame:04d}.png", bbox_inches="tight", dpi=500)
    plt.close(fig)


arguments = list(range(0, len(sim.snaps)))
func = partial(generate_frame, sim=sim)

print(len(arguments))
with Pool(processes=10) as pool:
    results = list(tqdm(pool.imap(func, arguments), total=len(arguments)))
