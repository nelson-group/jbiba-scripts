import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial import cKDTree
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit


def map_unstructured_to_structured_3d_optimized(positions, values, grid_size):
    """
    Map data from an unstructured 3D grid to a structured 3D grid using nearest neighbor interpolation.
    This version uses more efficient numpy array operations.

    Parameters:
    - positions (np.array): Array of shape (n, 3) containing the (x, y, z) positions of unstructured data.
    - values (np.array): Array of shape (n,) containing the values at the unstructured grid points.
    - grid_size (tuple): Dimensions of the structured grid (height, width, depth).

    Returns:
    - structured_grid (np.array): A 3D array representing the structured grid values.
    """
    # Define the grid to which we want to map the unstructured data
    x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
    y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
    z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])
    grid_height, grid_width, grid_depth = grid_size

    x_grid = np.linspace(x_min, x_max, grid_width)
    y_grid = np.linspace(y_min, y_max, grid_height)
    z_grid = np.linspace(z_min, z_max, grid_depth)
    
    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
    # Reshape the grid to match the shape of the output structured grid
    if len(values.shape) < 2:
        structured_grid = np.zeros((grid_height, grid_width, grid_depth))
    else:
        structured_grid = np.zeros((grid_height, grid_width, grid_depth, values.shape[-1]))

    # Create a k-d tree for fast nearest neighbor search
    tree = cKDTree(positions)

    # Create a 3D array of coordinates for the structured grid
    grid_coords = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T

    # Query the k-d tree for nearest neighbors
    _, idx = tree.query(grid_coords, k=1)

    # Map values to the structured grid
    if len(values.shape) < 2:
        structured_grid = values[idx].reshape(grid_height, grid_width, grid_depth)
    else:
        structured_grid = values[idx].reshape(grid_height, grid_width, grid_depth, values.shape[-1])

    return structured_grid


def measure_1d_power_spectrum_vector_field(v, boxsize=1.0, bins=200):
    """
    Computes the 1D power spectrum of a 3D vector field defined on a uniform grid.

    Parameters
    ----------
    v : ndarray, shape (Nx, Ny, Nz, 3)
        Real-space 3D vector field.
    boxsize : float or tuple
        Physical size of the domain (assumes cubic if float).
    bins : int or array
        Number of radial bins or custom bin edges (in k-space).

    Returns
    -------
    k_bin_centers : ndarray
        Centers of the k bins (in units of 1/boxsize).
    Pk : ndarray
        Power spectrum P(k) averaged in each shell.
    """
    assert v.ndim == 4 and v.shape[-1] == 3, "Field must be 3D vector on grid"

    Nx, Ny, Nz, _ = v.shape
    Lx, Ly, Lz = (boxsize, boxsize, boxsize) if np.isscalar(boxsize) else boxsize

    # Compute FFT of each component
    v_k = np.fft.fftn(v, norm="ortho")
    # v_k = np.fft.fftshift(v_k)  # center zero mode (optional)
    power = np.sum(np.abs(v_k)**2, axis=-1)  # sum over components, shape: (Nx,Ny,Nz)

    # Build k-space grid
    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2 * np.pi
    kz = np.fft.fftfreq(Nz, d=Lz / Nz) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.sqrt(kx**2 + ky**2 + kz**2).ravel()
    power = power.ravel()

    # Define bins in log space
    k_nonzero = k[k > 0]
    k_min = k_nonzero.min()
    k_max = k.max()

    if np.isscalar(bins):
        bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), bins + 1)
    else:
        bin_edges = np.asarray(bins)

    # Bin the power spectrum
    Pk, _, _ = binned_statistic(k, power, bins=bin_edges, statistic='mean')
    k_bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

    Pk *= k_bin_centers**2

    return k_bin_centers, Pk


# velocity power spectrum measurment

for 
    k_bins, Pk = measure_1d_power_spectrum_vector_field(structured_velocity)
    valid = ~np.isnan(Pk)
    Pk_avg[valid] = Pk_avg[valid] + Pk[valid]


Pk_avg = Pk_avg/num

nonzero = np.where(Pk_avg > 0)


cascade = np.where((k_bins[nonzero] > 20) & (k_bins[nonzero] < 60))

# Take logs
logk = np.log10(k_bins[nonzero][cascade])
logP = np.log10(Pk_avg[nonzero][cascade])

# Linear fit: logP = -alpha * logk + logA
coeffs = np.polyfit(logk, logP, 1)
slope, intercept = coeffs
alpha = -slope
A = 10**intercept

plt.plot(k_bins[nonzero], Pk_avg[nonzero])
plt.plot(k_bins[nonzero][cascade], A * k_bins[nonzero][cascade]**(-alpha), '--', label=f'Fit: $k^{{-{alpha:.2f}}}$')
plt.legend()
plt.xlabel("k")
plt.ylabel("P(k)")
# plt.grid(True, which="both")
plt.xscale("log")
plt.yscale("log")
plt.axvline(x=2*np.pi/1, color='r', linestyle='--', linewidth=1.5)
# plt.axvline(x=2*np.pi/1 * 128, color='r', linestyle='--', linewidth=1.5)
plt.axvline(x=2*np.pi/1 * grid, color='r', linestyle='--', linewidth=1.5)
plt.xlabel(r"$k[kpc^2]$")
plt.ylabel(r"$E(k)$")
plt.savefig(f"{sim}/velpowerspec_snap60-80.png")