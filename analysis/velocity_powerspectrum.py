import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial import cKDTree
import matplotlib.animation as animation
from scipy.optimize import curve_fit
import time

def map_unstructured_to_structured_3d_optimized(positions, values, grid_size, tree=None, workers=8):
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
    if tree == None:
        tree = cKDTree(positions)
    
    x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
    y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
    z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])
    grid_height, grid_width, grid_depth = grid_size

    x_grid = np.linspace(x_min, x_max, grid_width)
    y_grid = np.linspace(y_min, y_max, grid_height)
    z_grid = np.linspace(z_min, z_max, grid_depth)
    
    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    
    # Reshape the grid to match the shape of the output structured grid
    structured_grid = np.zeros((grid_height, grid_width, grid_depth))

    # Create a 3D array of coordinates for the structured grid
    grid_coords = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T

    # Query the k-d tree for nearest neighbors
    _, idx = tree.query(grid_coords, k=1, workers=workers)

    # Map values to the structured grid
    structured_grid = values[idx].reshape(grid_height, grid_width, grid_depth)

    return structured_grid


scale = 3.086e13


def radial_1D_power_spectrum(coords, velos, grid_size=200, scale=scale):
    tree = cKDTree(coords)
    grid_velocity_x = map_unstructured_to_structured_3d_optimized(coords, velos[:, 0], (grid_size, grid_size, grid_size), tree=tree)
    grid_velocity_y = map_unstructured_to_structured_3d_optimized(coords, velos[:, 1], (grid_size, grid_size, grid_size), tree=tree)
    grid_velocity_z = map_unstructured_to_structured_3d_optimized(coords, velos[:, 2], (grid_size, grid_size, grid_size), tree=tree)

    transformed_velocity_x = np.fft.fftn(grid_velocity_x)
    transformed_velocity_y = np.fft.fftn(grid_velocity_y)
    transformed_velocity_z = np.fft.fftn(grid_velocity_z)

    transformed_velocity_x_shift = np.fft.fftshift(transformed_velocity_x)
    transformed_velocity_y_shift = np.fft.fftshift(transformed_velocity_y)
    transformed_velocity_z_shift = np.fft.fftshift(transformed_velocity_z)


    power_spectrum3d = (transformed_velocity_x_shift.real**2 + transformed_velocity_x_shift.imag**2 
                        + transformed_velocity_y_shift.real**2 + transformed_velocity_y_shift.imag**2
                        + transformed_velocity_z_shift.real**2 + transformed_velocity_z_shift.imag**2)


    center = (grid_size//2, grid_size//2, grid_size//2)

    x_indices, y_indices, z_indices = np.indices(power_spectrum3d.shape)

    radius = np.sqrt((x_indices - center[0])**2 + (y_indices - center[1])**2 + (z_indices - center[2])**2)

    flat_radius = radius.flatten()
    flat_pspec = power_spectrum3d.flatten()

    num_bins = 1000
    bin_edges = np.logspace(np.log10(grid_size/2), 1, num_bins)[::-1]# np.linspace(np.min(flat_radius), np.max(flat_radius), num_bins + 1)

    # Compute the bin indices
    bin_indices = np.digitize(flat_radius, bin_edges)

    # Compute the average value per bin
    binned_averages = []
    bin_centers = []

    for i in range(1, len(bin_edges)):
        # Get indices of elements in the current bin
        bin_vals = flat_pspec[bin_indices == i]
        if len(bin_vals) > 0:
            # Calculate the average for the current bin
            binned_averages.append(np.mean(bin_vals))
            bin_centers.append(np.mean(bin_edges[i-1:i+1]))  # Bin center
    
    return np.array(bin_centers), np.array(bin_centers)**3 * binned_averages/scale


def average_power_spectrum(start, stop, grid_size=200, scale=scale, out="real-turbtest-128"):
    kek_vals_list = list()
    for i in range(start, stop):
        print(i)
        filename= f"{out}/snap_{i:03d}.hdf5" # str(sys.argv[1])
        with h5py.File(filename, 'r') as file:
            gas = file["PartType0"]
            velocities = np.array(gas["Velocities"])   
            coordinates = np.array(gas["Coordinates"])

        k_vals, kek_vals = radial_1D_power_spectrum(coordinates, velocities, grid_size=grid_size, scale=scale)
        kek_vals_list.append(kek_vals)
    kek_vals_list = np.array(kek_vals_list)
    kek_vals_mean = np.mean(kek_vals_list, axis=0)
    return k_vals, kek_vals_mean


def power_law(x, A):
    p = -2/3
    return A * x**p

fig, ax = plt.subplots()

cmap = plt.get_cmap('Spectral')

for j in range(50, 500, 100):
    print(j)
    k_vals, kek_vals = average_power_spectrum(j, j+100)
    ax.plot(k_vals, kek_vals, linewidth=1, color=cmap(j/500))


idx = np.where((10 < k_vals) & (k_vals < 25))[0]

x_data = k_vals[idx]
y_data = kek_vals[idx]

popt, pcov = curve_fit(power_law, x_data, y_data)

A_best = popt[0]

# Plot the results
ax.plot(np.linspace(10, 100), A_best * np.linspace(10, 100)**(-2/3), color="gray", linewidth=1)
ax.set_xlim((10, 100))
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_ylabel(r"$k E_{v}(k)$")
ax.set_xlabel(r"$k$")
ax.tick_params(
    axis="both",         
    which="both",      
    direction="in",      
    top=True,            
    bottom=True,         
    left=True,       
    right=True  
)
for spine in ax.spines.values():
    spine.set_linewidth(1.5)
fig.tight_layout()
fig.savefig("velocitypowerspectrum_differen_snaps.png")