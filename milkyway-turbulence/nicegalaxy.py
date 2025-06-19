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


os.environ['LATEX'] = '~/texlive/2025/bin/x86_64-linux/pdflatex'
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"""
        \usepackage{amsmath}
        \usepackage{bm}
        \usepackage{geometry}
    """,
    "font.family": "serif",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.linewidth": 1.5,       # Thicker border
    "xtick.major.width": 1.2,    # Thicker x ticks
    "ytick.major.width": 1.2,    # Thicker y ticks
    "xtick.labelsize": 14,       # Larger tick labels
    "ytick.labelsize": 14,
    "axes.labelsize": 16,        # Larger axis labels
    "axes.titlesize": 18,        # Larger title
    "lines.linewidth": 2.0
})


cutout_path = "/virgotng/universe/IllustrisTNG/TNG50-1/postprocessing/MWM31s/cutouts/snap_099"

galaxies_hdf5 = glob.glob(os.path.join(cutout_path, '*.hdf5'))
galaxies = [int((os.path.basename(f)).split(".")[0]) for f in galaxies_hdf5]


def measure_1d_power_spectrum_vector_field(v, boxsize=1.0, bins=200, spectral_weight=None):
    Nx, Ny, Nz, _ = v.shape
    Lx, Ly, Lz = (boxsize, boxsize, boxsize) if np.isscalar(boxsize) else boxsize

    v_k = np.fft.fftn(v, norm="ortho") 

    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2 * np.pi
    kz = np.fft.fftfreq(Nz, d=Lz / Nz) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = np.sqrt(kx**2 + ky**2 + kz**2)

    if isinstance(spectral_weight, float):
        k_dot_v = kx*v_k[:, :, :, 0] + ky*v_k[:, :, :, 1] + kz*v_k[:, :, :, 2]
        print(k_dot_v.shape, k_mag.shape)
        proj_v_k = np.zeros_like(v_k)
        proj_v_k[:, :, :, 0] = spectral_weight*v_k[:, :, :, 0] + (1 - 2*spectral_weight)*k_dot_v*kx/k_mag**2
        proj_v_k[:, :, :, 1] = spectral_weight*v_k[:, :, :, 1] + (1 - 2*spectral_weight)*k_dot_v*ky/k_mag**2
        proj_v_k[:, :, :, 2] = spectral_weight*v_k[:, :, :, 2] + (1 - 2*spectral_weight)*k_dot_v*kz/k_mag**2
        power = np.sum(np.abs(proj_v_k)**2, axis=-1)
    else:
        power = np.sum(np.abs(v_k)**2, axis=-1)
    k = k_mag.ravel()
    power = power.ravel()

    k_nonzero = k[k > 0]
    k_min = k_nonzero.min()
    k_max = k.max()

    if np.isscalar(bins):
        bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), bins + 1)
    else:
        bin_edges = np.asarray(bins)

    Pk, _, _ = binned_statistic(k, power, bins=bin_edges, statistic='mean')
    k_bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    Pk *= k_bin_centers**2

    return k_bin_centers, Pk


def map_unstructured_to_structured_3d_batched(positions, values, grid_size, workers=8, batches=64, bounds=None, disable=False):
    if bounds is None:
        x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
        y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
        z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])
    else:
        x_min, x_max = bounds[0][0], bounds[1][0]
        y_min, y_max = bounds[0][1], bounds[1][1]
        z_min, z_max = bounds[0][2], bounds[1][2]
    grid_height, grid_width, grid_depth = grid_size

    x_grid = np.linspace(x_min, x_max, grid_width)
    y_grid = np.linspace(y_min, y_max, grid_height)
    z_grid = np.linspace(z_min, z_max, grid_depth)
    
    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='xy')
    
    if len(values.shape) < 2:
        structured_grid = np.zeros((grid_height, grid_width, grid_depth))
    else:
        structured_grid = np.zeros((grid_height, grid_width, grid_depth, values.shape[-1]))

    tree = cKDTree(positions)

    grid_coords = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T

    idxs = []
    for i in tqdm(range(0, grid_coords.shape[0], grid_coords.shape[0]//batches), desc="Querying KDTree", disable=disable):
        batch = grid_coords[i:i+(grid_coords.shape[0]//batches)]
        _, idx = tree.query(batch, k=1, workers=workers)
        idxs.append(idx)

    if len(values.shape) < 2:
        structured_grid = values[idxs].reshape(grid_height, grid_width, grid_depth)
    else:
        structured_grid = values[idxs].reshape(grid_height, grid_width, grid_depth, values.shape[-1])

    return structured_grid


def map_unstructured_to_structured_3d_normal(positions, values, grid_size, bounds=None, disable=False):
    if bounds is None:
        x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
        y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
        z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])
    else:
        x_min, x_max = bounds[0][0], bounds[1][0]
        y_min, y_max = bounds[0][1], bounds[1][1]
        z_min, z_max = bounds[0][2], bounds[1][2]
    grid_height, grid_width, grid_depth = grid_size

    x_grid = np.linspace(x_min, x_max, grid_width)
    y_grid = np.linspace(y_min, y_max, grid_height)
    z_grid = np.linspace(z_min, z_max, grid_depth)
    
    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
    if len(values.shape) < 2:
        structured_grid = np.zeros((grid_height, grid_width, grid_depth))
    else:
        structured_grid = np.zeros((grid_height, grid_width, grid_depth, values.shape[-1]))

    tree = cKDTree(positions)

    grid_coords = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T

    _, idxs = tree.query(grid_coords, k=1)

    if len(values.shape) < 2:
        structured_grid = values[idxs].reshape(grid_height, grid_width, grid_depth)
    else:
        structured_grid = values[idxs].reshape(grid_height, grid_width, grid_depth, values.shape[-1])

    return structured_grid


def map_unstructured_to_structured_slice_optimized(positions, values, grid_size, grid, tree=None, workers=8):
    x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
    y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
    z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])

    grid_height, grid_width, grid_depth = tuple(map(len, grid))

    x_grid = np.linspace(x_min + grid[0][0]*(x_max - x_min)/grid_size, x_min + grid[0][-1]*(x_max - x_min)/grid_size, grid_height)
    y_grid = np.linspace(y_min + grid[1][0]*(y_max - y_min)/grid_size, y_min + grid[1][-1]*(y_max - y_min)/grid_size, grid_width)
    z_grid = np.linspace(z_min + grid[2][0]*(z_max - z_min)/grid_size, z_min + grid[2][-1]*(z_max - z_min)/grid_size, grid_depth)
    
    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='xy')
    
    if len(values.shape) < 2:
        structured_grid = np.zeros((grid_height, grid_width, grid_depth))
    else:
        structured_grid = np.zeros((grid_height, grid_width, grid_depth, values.shape[-1]))

    if not tree:
        tree = cKDTree(positions)

    grid_coords = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T

    _, idx = tree.query(grid_coords, k=1, workers=workers)

    if len(values.shape) < 2:
        structured_grid = values[idx].reshape(grid_height, grid_width, grid_depth)
    else:
        structured_grid = values[idx].reshape(grid_height, grid_width, grid_depth, values.shape[-1])

    if 1 in tuple(map(len, grid)):
        return np.squeeze(structured_grid)
    return structured_grid


def structured_column(positions, values, grid_size=200, workers=8, x_lim=None, y_lim=None, z_lim=None):
    if x_lim is not None:
        region_mask = ((x_lim[0] <= positions[:, 0]) & (positions[:, 0] <= x_lim[1]) 
                       & (y_lim[0] <= positions[:, 1]) & (positions[:, 1] <= y_lim[1]) 
                       & (z_lim[0] <= positions[:, 2]) & (positions[:, 2] <= z_lim[1]))
        positions = positions[region_mask]
        values = values[region_mask]
    tree = cKDTree(positions)
    struct_col = np.zeros((grid_size, grid_size))
    for i in tqdm(range(grid_size), desc="Slicing"):
        grid = (range(grid_size), range(grid_size), range(i, i + 1))
        mapped = map_unstructured_to_structured_slice_optimized(positions, values, grid_size=grid_size, grid=grid, tree=tree, workers=workers)
        struct_col = struct_col + mapped
    struct_col = struct_col/grid_size
    return struct_col


def power_spectrum_measurment():
    bins = 200
    fig, ax = plt.subplots()
    Pk_avg = np.zeros(bins)
    k_bins, Pk = measure_1d_power_spectrum_vector_field(struct_col, boxsize=541.1172579033118, bins=bins)
    valid = ~np.isnan(Pk)
    Pk_avg[valid] = Pk_avg[valid] + Pk[valid]


    nonzero = np.where(Pk_avg > 0)
    # cascade = np.where((k_bins[nonzero] > 20) & (k_bins[nonzero] < 60))

    # logk = np.log10(k_bins[nonzero][cascade])
    # logP = np.log10(Pk_avg[nonzero][cascade])

    # # Linear fit: logP = -alpha * logk + logA
    # coeffs = np.polyfit(logk, logP, 1)
    # slope, intercept = coeffs
    # alpha = -slope
    # A = 10**intercept
    plot = ax.plot(k_bins[nonzero], Pk_avg[nonzero])
    # fit = ax.plot(k_bins[nonzero][cascade], A * k_bins[nonzero][cascade]**(-alpha), '--', label=f'Fit: $k^{{-{alpha:.2f}}}$')
    ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel("P(k)")
    ax.set_xscale("log")
    ax.set_yscale("log")


def compressive_ratio_measurment1(velocities, density, masses, internal_energy, bulk_v=None):
    volumes = masses/density
    mean_dens = np.sum(volumes*density)/np.sum(volumes)
    dens_var = np.sqrt(np.sum(volumes*density**2)/np.sum(volumes) - mean_dens**2)
    if bulk_v is None:
        bulk_v = np.mean(velocities, axis=0)
    soundspeed = np.sqrt(5/3 * (5/3 - 1) * internal_energy)
    avg_soundspeed = np.sum(volumes*soundspeed)/np.sum(volumes)
    rms_mach = np.sqrt(np.sum(volumes*np.sum(((velocities - bulk_v)**2), axis=-1)/soundspeed**2)/np.sum(volumes))
    # rms_mach = np.sqrt(np.sum(volumes*np.sum(((velocities - bulk_v)**2), axis=-1))/np.sum(volumes))/avg_soundspeed
    b1 = (1/rms_mach) * dens_var/mean_dens
    return b1, rms_mach, dens_var, mean_dens, avg_soundspeed


def compressive_ratio_measurment1_1(positions, velocities, density, internal_energy, masses, bounds, bulk_v=None):
    grid_density = map_unstructured_to_structured_3d_batched(positions, density, grid_size=(100, 100, 100), batches=1, bounds=bounds, disable=True)
    soundspeed = np.sqrt(5/3 * (5/3 - 1) * internal_energy)
    mach_sq = np.sum((velocities - np.mean(velocities, axis=0))**2, axis=-1)/soundspeed**2
    grid_mach_sq = map_unstructured_to_structured_3d_batched(positions, mach_sq, grid_size=(100, 100, 100), batches=1, bounds=bounds, disable=True)
    mean_dens = np.mean(grid_density)
    dens_var = np.std(grid_density)
    rms_mach = np.sqrt(np.mean(grid_mach_sq))
    radii = (masses/density)**(1/3)
    uniqueness = np.min(radii)/np.max(radii) # len(np.unique(grid_density.ravel()))/100**3
    b1_1 = (1/rms_mach) * dens_var/mean_dens
    return b1_1, rms_mach, dens_var, mean_dens, uniqueness


def compressive_ratio_measurment2(positions, velocities, density, masses, bounds):
    vols = masses/density
    # diams = 2 * (3/(4*np.pi) * vols)**(1/3)
    # grid_size = int((bounds[1][0] - bounds[0][0])/np.min(diams))
    grid_size = int((len(positions))**(1/3))
    grid_velocites = map_unstructured_to_structured_3d_batched(positions, velocities - np.mean(velocities, axis=0), grid_size=(grid_size, grid_size, grid_size), batches=1, bounds=bounds, disable=True)
    field_k = np.fft.fftn(grid_velocites, axes=(0, 1, 2))
    kx = np.fft.fftfreq(grid_size)
    ky = np.fft.fftfreq(grid_size)
    kz = np.fft.fftfreq(grid_size)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    K = np.stack((KX, KY, KZ), axis=-1)

    k_squared = np.sum(K**2, axis=-1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        k_unit = np.divide(K, k_squared, where=(k_squared != 0))

    k_dot_v = np.sum(field_k * np.conj(K), axis=-1, keepdims=True)
    v_comp_k = k_dot_v * k_unit
    v_comp_k[np.isnan(v_comp_k)] = 0 

    total_power_long = np.sum(np.abs(v_comp_k)**2)
    total_power = np.sum(np.abs(field_k)**2)
    return np.sqrt(3)*total_power_long/total_power


class Galaxy:
    def __init__(self, id):
        self.galay_path = f"/virgotng/universe/IllustrisTNG/TNG50-1/postprocessing/MWM31s/cutouts/snap_099/{id}.hdf5"

        self.gas = dict()
        self.stars = dict()
        self.bhs = dict()
        
        with h5py.File(self.galay_path, 'r') as file:
            gas_file = file["PartType0"]
            for key in gas_file.keys():
                self.gas[key] = np.array(gas_file[key])
            stars_file = file["PartType4"]
            for key in stars_file.keys():
                self.stars[key] = np.array(stars_file[key])
            bhs_file = file["PartType5"]
            for key in bhs_file.keys():
                self.bhs[key] = np.array(bhs_file[key])


        # self.center = np.array([0.5*(self.gas["Coordinates"][:, 0].max() + self.gas["Coordinates"][:, 0].min()), 
        #                         0.5*(self.gas["Coordinates"][:, 1].max() + self.gas["Coordinates"][:, 1].min()), 
        #                         0.5*(self.gas["Coordinates"][:, 2].max() + self.gas["Coordinates"][:, 2].min())])
        self.center = self.bhs["Coordinates"][0]
        self.disk_lim = np.array([[self.center[i] - 50, self.center[i] + 50] for i in range(3)])


        self.centered_coords = self.gas["Coordinates"] - self.center
        self.disk_mask = ((self.disk_lim[0][0] <= self.gas["Coordinates"][:, 0]) & (self.gas["Coordinates"][:, 0] <= self.disk_lim[0][1]) 
               & (self.disk_lim[1][0] <= self.gas["Coordinates"][:, 1]) & (self.gas["Coordinates"][:, 1] <= self.disk_lim[1][1]) 
               & (self.disk_lim[2][0] <= self.gas["Coordinates"][:, 2]) & (self.gas["Coordinates"][:, 2] <= self.disk_lim[2][1]))
        
        self.I = np.zeros((3, 3))
        x, y, z = self.stars["Coordinates"].T

        I_xx = np.sum(y**2 + z**2)
        I_yy = np.sum(x**2 + z**2)
        I_zz = np.sum(x**2 + y**2)
        I_xy = -np.sum(x * y)
        I_xz = -np.sum(x * z)
        I_yz = -np.sum(y * z)

        self.I = np.array([
            [I_xx, I_xy, I_xz],
            [I_xy, I_yy, I_yz],
            [I_xz, I_yz, I_zz]
        ])
        eigvals, eigvecs = np.linalg.eigh(self.I)
        min_index = np.argmin(eigvals)
        self.normal_vector = eigvecs[:, min_index]
        v = np.cross(self.normal_vector/np.linalg.norm(self.normal_vector), np.array([0, 0, 1]))
        c = np.dot(self.normal_vector/np.linalg.norm(self.normal_vector), np.array([0, 0, 1]))
        s = np.linalg.norm(v)
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        self.rotation = np.eye(3) + vx + vx @ vx * ((1 - c) / (s**2))

        self.rotated_gas_coords = self.centered_coords @ self.rotation.T

    
    def subdivide_box(self, indices, threshold, bounds, level=0): 
        if len(indices) < threshold:
            return None
        # b1, rms_mach, dens_var, mean_dens, avg_soundspeed = compressive_ratio_measurment1(self.gas["Velocities"][indices], self.gas["Density"][indices], 
        #                                        self.gas["Masses"][indices], self.gas["InternalEnergy"][indices])
        b1, rms_mach, dens_var, mean_dens, avg_soundspeed = compressive_ratio_measurment1_1(self.gas["Coordinates"][indices], self.gas["Velocities"][indices], 
                                                                                            self.gas["Density"][indices], self.gas["InternalEnergy"][indices], self.gas["Masses"][indices], bounds=bounds)
        b2 = compressive_ratio_measurment2(self.gas["Coordinates"][indices], self.gas["Velocities"][indices], self.gas["Density"][indices], self.gas["Masses"][indices], bounds=bounds)
        result = {
            "level": level,
            "b1": b1, 
            "rms_mach": rms_mach,
            "dens_var": dens_var, 
            "mean_dens": mean_dens,
            "c0": avg_soundspeed,
            "b2": b2,
            "bounds": bounds, 
            "nums": len(indices),
            "sfr": np.mean(self.gas["StarFormationRate"][indices])
        }

        (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
        xmid = (xmin + xmax) / 2
        ymid = (ymin + ymax) / 2
        zmid = (zmin + zmax) / 2

        sub_boxes = [
            ((x0, y0, z0), (x1, y1, z1))
            for x0, x1 in [(xmin, xmid), (xmid, xmax)]
            for y0, y1 in [(ymin, ymid), (ymid, ymax)]
            for z0, z1 in [(zmin, zmid), (zmid, zmax)]
        ]

        results = [result]
        for bmin, bmax in sub_boxes:
            in_box = (
                (self.gas["Coordinates"][indices, 0] >= bmin[0]) & (self.gas["Coordinates"][indices, 0] < bmax[0]) &
                (self.gas["Coordinates"][indices, 1] >= bmin[1]) & (self.gas["Coordinates"][indices, 1] < bmax[1]) &
                (self.gas["Coordinates"][indices, 2] >= bmin[2]) & (self.gas["Coordinates"][indices, 2] < bmax[2])
            )
            sub_indices = indices[in_box]
            if len(sub_indices) > 0:
                result = self.subdivide_box(sub_indices, threshold, (bmin, bmax), level + 1)
                if result is not None:
                    results.append(result)
        return results
