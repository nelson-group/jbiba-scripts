import nicegalaxy
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from collections import defaultdict
import h5py


galaxy = nicegalaxy.Galaxy(nicegalaxy.galaxies[30])
vor = Voronoi(galaxy.gas["Coordinates"])
neighbors = defaultdict(set)
for p1, p2 in vor.ridge_points:
    neighbors[p1].add(p2)
    neighbors[p2].add(p1)

print("Done with the voronoi.")


def get_cell_volumes(vor, start, n):
    volumes = []
    for i in range(start, n):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if -1 in region or len(region) == 0:
            continue
        try:
            verts = np.array([vor.vertices[j] for j in region])
            # if len(verts) < 4:
            #     continue
            hull = ConvexHull(verts)
            volumes.append(hull.volume)
        except:
            volumes.append(0)
    return volumes


# @njit
def mirrored_points_stable(points):
    lims = np.array([
        [points[:, 0].min(), points[:, 0].max()],
        [points[:, 1].min(), points[:, 1].max()],
        [points[:, 2].min(), points[:, 2].max()]
    ])
    N = len(points)
    mirr = np.zeros((6 * N, 3), dtype=points.dtype)
    for i in range(3):
        start = 2 * i * N
        mirr[start:start+N] = points
        mirr[start+N:start+2*N] = points
        mirr[start:start+N, i] = 2 * lims[i, 0] - points[:, i]
        mirr[start+N:start+2*N, i] = 2 * lims[i, 1] - points[:, i]
    return mirr 


refined_quantities = ["Masses", "InternalEnergy", "Density", "Velocities", "ElectronAbundance", "Machnumber", "MagneticField"]


def save_refined(gas, new_gas, active, level, res, error):
    refined_gas = dict()
    refined_gas["ParticleIDs"] = gas["ParticleIDs"][active]
    for key in new_gas:
        refined_gas[key] = new_gas[key][active]
        refined_gas[f"{key}_old"] = gas[key][active]
    with h5py.File(f"data/derefined/better/{level:03d}.h5", "w") as f:
        f.create_dataset("DerefinementLevel", data=level)
        f.create_dataset("DerefinementRes", data=res)
        f.create_dataset("CumVolError", data=error)
        f.create_dataset("ParticlesIDs", data=gas["ParticleIDs"][active])
        for key, array in new_gas.items():
            f.create_dataset(key, data=array[active])
            f.create_dataset(f"{key}_old", data=gas[key][active])


def derefine_queue(neighbors, vor, positions, vols, gas, maxres=-1):  
    new_gas = dict()
    for key in refined_quantities:
        new_gas[key] = gas[key].copy()

    refinement_stages = np.linspace(vols.min(), vols.max(), 100)
    current_stage_level = 0
    
    if maxres < 0:
        maxres = np.max(vols)

    mask2 = vols < maxres
    active = np.ones(len(positions), dtype=np.bool)
    active[~mask2] = 0
    subset_indices = np.where(active)[0]
    queue = sorted(subset_indices, key=lambda x: vols[x])

    count = 0
    error = 0
    for i, idx in enumerate(queue):
        if vols[idx] >= maxres:
            continue
        region = vor.regions[vor.point_region[idx]]
        if (-1 in region) or len(region) == 0:
            continue
        active[idx] = 0

        if vols[idx] > refinement_stages[current_stage_level]:
            print(f"Saving derefinement level {current_stage_level}. Cells are bigger than {refinement_stages[current_stage_level]}")
            print(f"Hopefully {(i/len(positions) * 100):.2f} % done. Current error is {error}")
            save_refined(gas, new_gas, active, current_stage_level, refinement_stages[current_stage_level], error)
            current_stage_level += 1


        neigh = neighbors[idx]
        delta_v = 0
        
        neigh_list = list(neigh)
        points_with_mirr = np.zeros((7*len(neigh) + 1, 3))
        points_with_mirr[1:(len(neigh) + 1)] = positions[neigh_list]
        points_with_mirr[0] = positions[idx]
        points_with_mirr[(len(neigh) + 1):] = mirrored_points_stable(points_with_mirr[1:(len(neigh) + 1)])

        try:
            vor_with = Voronoi(points_with_mirr)
            vor_without = Voronoi(points_with_mirr[1:])
        except Exception as e:
            continue

        vols_with = get_cell_volumes(vor_with, 0, len(neigh) + 1)
        vols_without = get_cell_volumes(vor_without, 0, len(neigh))


        for i in range(len(vols_without)):
            dv = vols_without[i] - vols_with[i + 1]
            frac = (vols_without[i] - vols_with[i + 1])/vols_with[0]
            new_mass = new_gas["Masses"][neigh_list[i]] + frac*new_gas["Masses"][idx]
            new_gas["InternalEnergy"][neigh_list[i]] = (frac*new_gas["Masses"][idx]*new_gas["InternalEnergy"][idx] + new_gas["InternalEnergy"][neigh_list[i]]*new_gas["Masses"][neigh_list[i]])/new_mass
            new_gas["Velocities"][neigh_list[i]] = (frac*new_gas["Masses"][idx]*new_gas["Velocities"][idx] + new_gas["Velocities"][neigh_list[i]]*new_gas["Masses"][neigh_list[i]])/new_mass
            new_gas["Density"][neigh_list[i]] = new_mass/(vols[neigh_list[i]] + dv)
            new_gas["ElectronAbundance"][neigh_list[i]] = (frac*new_gas["Masses"][idx] * new_gas["ElectronAbundance"][idx] + new_gas["Masses"][neigh_list[i]] * new_gas["ElectronAbundance"][neigh_list[i]])/new_mass
            new_gas["Machnumber"][neigh_list[i]] = (dv * new_gas["Machnumber"][idx] + vols[neigh_list[i]] * new_gas["Machnumber"][neigh_list[i]])/(vols[neigh_list[i]] + dv)
            new_gas["MagneticField"][neigh_list[i]] = (dv * new_gas["MagneticField"][idx] + vols[neigh_list[i]] * new_gas["MagneticField"][neigh_list[i]])/(vols[neigh_list[i]] + dv)
            new_gas["Masses"][neigh_list[i]] = new_mass
            vols[neigh_list[i]] += dv
            delta_v += dv


        error += np.abs(delta_v - vols[idx])
        count += 1
        if count > 100:
            break
        for neigh_idx in neigh_list:
            neighbors[neigh_idx] = neighbors[neigh_idx].union(neighbors[idx] - {neigh_idx})
            neighbors[neigh_idx].remove(idx)
    
    refined_gas = dict()
    refined_gas["ParticleIDs"] = gas["ParticleIDs"][active]
    for key in new_gas:
        refined_gas[key] = new_gas[key][active]
        refined_gas[f"{key}_old"] = gas[key][active]
    
    save_refined(gas, new_gas, active, len(refinement_stages), maxres, error)
    return refined_gas, error


derefine_queue(neighbors, vor, galaxy.gas["Coordinates"], galaxy.gas["Masses"]/galaxy.gas["Density"], galaxy.gas)
