import numpy as np
import matplotlib.pyplot as plt
import h5py
from scipy.spatial import cKDTree
import matplotlib.animation as animation
import sys  
from mpl_toolkits.axes_grid1 import make_axes_locatable


def map_unstructured_to_structured_3d(positions, values, grid_size):
    """
    Map data from an unstructured 3D grid to a structured 3D grid using nearest neighbor interpolation.

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
    
    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid)
    structured_grid = np.zeros((grid_height, grid_width, grid_depth))

    # Create a k-d tree for fast nearest neighbor search
    tree = cKDTree(positions)

    # For each point in the structured grid, find the nearest point in the unstructured grid
    for i in range(grid_height):
        for j in range(grid_width):
            for k in range(grid_depth):
                # Find the nearest unstructured grid point for the current grid cell
                dist, idx = tree.query([X_grid[i, j, k], Y_grid[i, j, k], Z_grid[i, j, k]], k=1)
                structured_grid[i, j, k] = values[idx]  # Assign the value of the nearest neighbor

    return structured_grid


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
    structured_grid = np.zeros((grid_height, grid_width, grid_depth))

    # Create a k-d tree for fast nearest neighbor search
    tree = cKDTree(positions)

    # Create a 3D array of coordinates for the structured grid
    grid_coords = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T

    # Query the k-d tree for nearest neighbors
    _, idx = tree.query(grid_coords, k=1)

    # Map values to the structured grid
    structured_grid = values[idx].reshape(grid_height, grid_width, grid_depth)

    return structured_grid


sims = ["driving128-nonsol", "driving64", "driving32", "driving128-mach10", "driving128-mach10-forceramp"]
sim = sims[4]
for i in range(0, 221):
    filename= f"{sim}/output/snap_{i:03d}.hdf5" # str(sys.argv[1])
    print(filename)
    with h5py.File(filename, 'r') as file:
        gas = file["PartType0"]
        coordinates = np.array(gas["Coordinates"])
        velocities = np.array(gas["Velocities"])   
        dens = np.array(gas["Density"])
        acceleration = np.array(gas["Acceleration"])


    N = 400
    structured_grid_dens = map_unstructured_to_structured_3d_optimized(coordinates, np.sqrt(np.sum(velocities**2, axis=1)), (N, N, N))
    # structured_grid_vel = map_unstructured_to_structured_3d_optimized(coordinates, np.sqrt(np.sum(velocities**2, axis=1)), (N, N, N))
    # col_dens = np.sum(structured_grid_dens, axis=0)/N
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(np.log10(structured_grid_dens[:, :, N//2]), cmap='plasma', origin='lower', vmin=1.5, vmax=3.5)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax, label=r"$log(|v|) [km/s]$") # label=r'$ log() [M_{\odot}/kpc^3]$ ') # , label=r'$ \rho [M_{\odot}/kpc^3]$ ')

    ax.set_xticks([])
    ax.set_yticks([])
    fig.savefig(f"{sim}/plots/vel/vel_{filename.split("/")[-1].split(".")[0]}.png")
    plt.close()