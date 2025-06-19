import numpy as np
import nicegalaxy
import h5py
from numba import njit, prange
from scipy.spatial import cKDTree
import time, sys
from numba.typed import List
from tqdm import tqdm


def update_progress(fraction_done, bar_length=40):
    percent = int(fraction_done * 100)
    filled_len = int(bar_length * fraction_done)
    bar = '=' * filled_len + '-' * (bar_length - filled_len)
    sys.stdout.write(f'\r[{bar}] {percent:3d}%')
    sys.stdout.flush()


@njit
def init_mean_vels(N, velx, vely, velz, inds, previous_turbulentx, previous_turbulenty, previous_turbulentz, current_width, cell_radii):
    for i in range(N):
        bulkx = np.mean(velx[inds[i]])
        bulky = np.mean(vely[inds[i]])
        bulkz = np.mean(velz[inds[i]])
        previous_turbulentx[i] = velx[i] - bulkx
        previous_turbulenty[i] = vely[i] - bulky
        previous_turbulentz[i] = velz[i] - bulkz
        current_width[i] = max(current_width[i] + cell_radii[i], 1.05 * current_width[i])


@njit
def calculate_mean_vels(N, velx, vely, velz, inds, previous_turbulentx, previous_turbulenty, previous_turbulentz, current_width, cell_radii, shocks, coherence_length):
    for i in range(N):
        if len(inds[i]) < 2:
            continue
        bulkx = np.mean(velx[inds[i]])
        bulky = np.mean(vely[inds[i]])
        bulkz = np.mean(velz[inds[i]])

        delta = np.max(np.abs(np.array([velx[i] - bulkx, vely[i] - bulky, velz[i] - bulkz]) - 1))
        shocked = False
        for l in inds[i]:
            if shocks[l] > 1.3:
                shocked = True
                # print("im shocked")
                break
        if (delta < 0.05) or shocked:
            coherence_length[i] = current_width[i]
            current_width[i] = 0.0
        else:
            current_width[i] = max(current_width[i] + cell_radii[i], 1.05 * current_width[i])

        previous_turbulentx[i] = velx[i] - bulkx
        previous_turbulenty[i] = vely[i] - bulky
        previous_turbulentz[i] = velz[i] - bulkz


def box_smoothing_velocity_decomposition(positions, velocities, shocks):
    N = positions.shape[0]
    vols = np.zeros_like(shocks) + 91.6
    cell_radii = (3/(4*np.pi) * vols)**(1/3)
    current_width = 3 * cell_radii
    velx = velocities[:, 0]
    vely = velocities[:, 1]
    velz = velocities[:, 2]
    previous_turbulentx = np.zeros_like(velx)
    previous_turbulenty = np.zeros_like(velx)
    previous_turbulentz = np.zeros_like(velx)
    coherence_scale = np.zeros(len(positions))

    tree = cKDTree(positions)
    inds = tree.query_ball_point(positions, current_width, workers=8)
    numba_inds = List()
    for ind_list in inds:
        numba_inds.append(np.array(ind_list))

    init_mean_vels(N, velx, vely, velz, numba_inds, previous_turbulentx, previous_turbulenty, previous_turbulentz, current_width, cell_radii)

    while np.sum(current_width) > 0:
        update_progress(np.sum(current_width == 0.0)/len(positions))
        print(np.mean(current_width))
        inds = tree.query_ball_point(positions, current_width, workers=8)
        numba_inds = List()
        for ind_list in inds:
            numba_inds.append(np.array(ind_list))
        calculate_mean_vels(N, velx, vely, velz, numba_inds, previous_turbulentx, previous_turbulenty, previous_turbulentz, current_width, cell_radii, shocks, coherence_scale)
    
    previous_turbulent = np.column_stack((previous_turbulentx, previous_turbulenty, previous_turbulentz))
    return velocities - previous_turbulent, previous_turbulent, coherence_scale


dens = np.load("refined_dens.npy")
vels = np.load("refined_vels.npy")
galaxy = nicegalaxy.Galaxy(nicegalaxy.galaxies[30])
flat_vels = vels.reshape(-1, 3)
binsx = np.linspace(galaxy.gas["Coordinates"][:, 0].min(), galaxy.gas["Coordinates"][:, 0].max(), 120 + 1)
binsy = np.linspace(galaxy.gas["Coordinates"][:, 1].min(), galaxy.gas["Coordinates"][:, 2].max(), 120 + 1)
binsz = np.linspace(galaxy.gas["Coordinates"][:, 2].min(), galaxy.gas["Coordinates"][:, 1].max(), 120 + 1)
x = 0.5 * (binsx[:-1] + binsx[1:])
y = 0.5 * (binsy[:-1] + binsy[1:])
z = 0.5 * (binsz[:-1] + binsz[1:])

X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # 'ij' for matrix-style indexing
grid = np.stack((X, Y, Z), axis=-1)
coords = grid.reshape(-1, 3)

turb = box_smoothing_velocity_decomposition(coords, flat_vels, np.zeros(120**3))

with h5py.File(f'data/derefined-vortexp-{nicegalaxy.galaxies[30]}.h5', 'w') as f:
    f.create_dataset("TurbVelocities", data=turb[1])
    f.create_dataset("CoherenceScale", data=turb[2])
