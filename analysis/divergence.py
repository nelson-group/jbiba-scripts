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


sims = ["driving128-nonsol", "driving64", "driving32", "driving128-mach10", "driving128-mach10-forceramp"]
sim = sims[4]

grid = 100

i = 10
filename = f"{sim}/output/snap_{i:03d}.hdf5" # str(sys.argv[1])
with h5py.File(filename, 'r') as file:
    gas = file["PartType0"]
    coordinates = np.array(gas["Coordinates"])
    velocities = np.array(gas["Velocities"])   
    density = np.array(gas["Density"])
    # pressure = np.array(gas["Pressure"])
    acc = np.array(gas["Acceleration"])
    # mass = np.array(gas["Masses"])


structured_acc = map_unstructured_to_structured_3d_optimized(coordinates, velocities, (grid, grid, grid))


dx = 1/grid
vx = structured_acc[:, :, :, 0]
dvx_dx = (np.roll(structured_acc[:, :, :, 0], -1, axis=0) - np.roll(structured_acc[:, :, :, 0], 1, axis=0)) / (2 * dx)
dvy_dy = (np.roll(structured_acc[:, :, :, 1], -1, axis=1) - np.roll(structured_acc[:, :, :, 1], 1, axis=1)) / (2 * dx)
dvz_dz = (np.roll(structured_acc[:, :, :, 2], -1, axis=2) - np.roll(structured_acc[:, :, :, 2], 1, axis=2)) / (2 * dx)

div = dvx_dx + dvy_dy + dvz_dz

col_div = np.sum(np.abs(div), axis=2)

plt.imshow(div[:, :, grid//2])
plt.colorbar()
plt.savefig(f"{sim}/plots/acc_div.png")