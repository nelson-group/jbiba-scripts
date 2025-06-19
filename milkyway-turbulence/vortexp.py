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


def box_smoothing_velocity_decomposition(positions, velocities, masses, density, shocks):
    N = positions.shape[0]
    vols = masses/density
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
        inds = tree.query_ball_point(positions, current_width, workers=8)
        numba_inds = List()
        for ind_list in inds:
            numba_inds.append(np.array(ind_list))
        calculate_mean_vels(N, velx, vely, velz, numba_inds, previous_turbulentx, previous_turbulenty, previous_turbulentz, current_width, cell_radii, shocks, coherence_scale)
    
    previous_turbulent = np.column_stack((previous_turbulentx, previous_turbulenty, previous_turbulentz))
    return velocities - previous_turbulent, previous_turbulent, coherence_scale


for galaxy_idx in tqdm(range(50), desc="Looping over Galaxies"):
    galaxy = nicegalaxy.Galaxy(nicegalaxy.galaxies[galaxy_idx])

    turb = box_smoothing_velocity_decomposition(galaxy.gas["Coordinates"], galaxy.gas["MagneticField"], galaxy.gas["Masses"], 
                                            galaxy.gas["Density"], galaxy.gas["Machnumber"])
    print(" \n")
    with h5py.File(f'data/turb-magnetic/{nicegalaxy.galaxies[galaxy_idx]}.h5', 'w') as f:
        f.create_dataset("TurbMagneticField", data=turb[1])
        f.create_dataset("CoherenceScale", data=turb[2])