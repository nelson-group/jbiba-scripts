import nicegalaxy
import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.optimize import linprog
from scipy.spatial import QhullError
from itertools import batched
from functools import partial


galaxy = nicegalaxy.Galaxy(nicegalaxy.galaxies[30])


def mirrored_points_frame(points, center, frac=0.1):
    lims = np.array([
        [points[:, 0].min(), points[:, 0].max()],
        [points[:, 1].min(), points[:, 1].max()],
        [points[:, 2].min(), points[:, 2].max()]
    ])

    inner = (lims[0, 1] - lims[0, 0]) * (1 - frac)/2
    in_middle = np.max(np.abs(points - center), axis=-1) < inner
    frame_points = points[~in_middle]
    # frame_density = density[~in_middle]
    N = len(frame_points)
    mirr = np.zeros((6 * N, 3), dtype=points.dtype)
    # dens = np.zeros((6 * N), dtype=points.dtype)
    for i in range(3):
        start = 2 * i * N
        mirr[start:start+N] = frame_points
        mirr[start+N:start+2*N] = frame_points
        mirr[start:start+N, i] = 2 * lims[i, 0] - frame_points[:, i]
        mirr[start+N:start+2*N, i] = 2 * lims[i, 1] - frame_points[:, i]
        # dens[start:start+N] = frame_density
        # dens[start+N:start+2*N] = frame_density

    cropped = mirr[np.max(np.abs(mirr - center), axis=-1) < (((lims[0, 1] - lims[0, 0]))*(1 + frac)/2)]
    # cropped_dens = dens[np.max(np.abs(mirr - center), axis=-1) < (((lims[0, 1] - lims[0, 0]))*(1 + frac)/2)]
    return np.concatenate((points, cropped))


points_with_mirr = mirrored_points_frame(galaxy.gas["Coordinates"], galaxy.bhs["Coordinates"][0])
vor_with_mirr = Voronoi(points_with_mirr)


def get_cell_volumes(vor, start, n):
    volumes = []
    for i in range(start, n):
        region_index = vor.point_region[i]
        region = vor.regions[region_index]
        if -1 in region or len(region) == 0:
            continue
        try:
            verts = np.array([vor.vertices[j] for j in region])
            hull = ConvexHull(verts)
            volumes.append(hull.volume)
        except:
            volumes.append(0)
    return volumes


def cube_halfspaces(xmin, xmax, ymin, ymax, zmin, zmax):
    return np.array([
        [-1, 0, 0, xmin],  
        [1, 0, 0, -xmax],  
        [0, -1, 0, ymin], 
        [0, 1, 0, -ymax],  
        [0, 0, -1, zmin],
        [0, 0, 1, -zmax]  
    ])


def distribute(density, vols, binsx, binsy, binsz, grid_size, gas_is, vertss): # range(len(gas["Coordinates"])):
        derefined_mass = np.zeros((grid_size, grid_size, grid_size))
        errors = np.zeros(2)
        for gas_i, verts in zip(gas_is, vertss):
            # region_index = vor.point_region[gas_i]
            # region = vor.regions[region_index]
            # if -1 in region or len(region) == 0:
            #     return
            # verts = np.array([vor.vertices[j] for j in region])

            xlim = (verts[:, 0].min(), verts[:, 0].max())
            ylim = (verts[:, 1].min(), verts[:, 1].max())
            zlim = (verts[:, 2].min(), verts[:, 2].max())
        
            idxxlim = (np.ceil((xlim[0] - binsx[0])/(binsx[-1] - binsx[0]) * grid_size), np.ceil((xlim[1] - binsx[0])/(binsx[-1] - binsx[0]) * grid_size))
            idxylim = (np.ceil((ylim[0] - binsy[0])/(binsy[-1] - binsy[0]) * grid_size), np.ceil((ylim[1] - binsy[0])/(binsy[-1] - binsy[0]) * grid_size))
            idxzlim = (np.ceil((zlim[0] - binsz[0])/(binsz[-1] - binsz[0]) * grid_size), np.ceil((zlim[1] - binsz[0])/(binsz[-1] - binsz[0]) * grid_size))

            vol_tot = 0

            for i in range(max(int(idxxlim[0]), 1), int(idxxlim[1]) + 1):
                for j in range(max(int(idxylim[0]), 1), int(idxylim[1]) + 1):
                    for k in range(max(int(idxzlim[0]), 1), int(idxzlim[1]) + 1):
                        # binsx[i - 1], binsx[i]
                        hull = ConvexHull(verts)

                        poly_halfspaces = []
                        for simplex in hull.equations:
                            normal = simplex[:-1]
                            offset = simplex[-1]
                            poly_halfspaces.append(np.append(normal, offset))

                        poly_halfspaces = np.array(poly_halfspaces)
                        hs = np.vstack((cube_halfspaces(binsx[i - 1], binsx[i], binsy[j - 1], binsy[j], binsz[k - 1], binsz[k]), poly_halfspaces))

                        norm_vector = np.reshape(np.linalg.norm(hs[:, :-1], axis=1), (hs.shape[0], 1))
                        c = np.zeros((hs.shape[1],))
                        c[-1] = -1
                        A = np.hstack((hs[:, :-1], norm_vector))
                        b = - hs[:, -1:]
                        res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
                        interior_point = res.x[:-1]

                        try:
                            hs_int = HalfspaceIntersection(hs, interior_point)
                        except QhullError:
                            # print("QhullError. Moving on")
                            errors[1] += 1
                            continue

                        intersection_hull = ConvexHull(hs_int.intersections)
                        volume = intersection_hull.volume
                        vol_tot += volume
                        derefined_mass[i - 1, j - 1, k - 1] += volume*density[gas_i]
            errors[0] += np.abs(vol_tot - vols[gas_i])
        return derefined_mass, errors


def derefine(vor, gas, minvol=-1):
    vols = gas["Masses"]/gas["Density"]
    if minvol < 0:
        minvol = np.mean(vols)
    grid_size = int((gas["Coordinates"][:, 0].max() - gas["Coordinates"][:, 0].min())/(minvol**(1/3)))

    binsx = np.linspace(gas["Coordinates"][:, 0].min(), gas["Coordinates"][:, 0].max(), grid_size + 1)
    binsx[0] -= 1
    binsy = np.linspace(gas["Coordinates"][:, 1].min(), gas["Coordinates"][:, 1].max(), grid_size + 1)
    binsy[0] -= 1
    binsz = np.linspace(gas["Coordinates"][:, 2].min(), gas["Coordinates"][:, 2].max(), grid_size + 1)
    binsz[0] -= 1
    # idxx = np.digitize(gas["Coordinates"][:, 0], binsx, right=True)
    # idxy = np.digitize(gas["Coordinates"][:, 1], binsy, right=True)
    # idxz = np.digitize(gas["Coordinates"][:, 2], binsz, right=True)
    binsx[0] += 1
    binsy[0] += 1
    binsz[0] += 1

    tasks = list(range(len(gas["Coordinates"])))
    get_verts = lambda batch: [np.array([vor.vertices[j] for j in region]) for region in [vor.regions[vor.point_region[i]] for i in batch]]
    batched_tasks = [(i_group, get_verts(i_group)) for i_group in batched(tasks, 10000)]

    thread_results = []
    # with concurrent.futures.ThreadPoolExecutor() as executor:
    #     futures = [executor.submit(distribute, i_group) for i_group in batched(tasks, 100)]
    #     for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
    #         thread_results.append(future.result())

    distr = partial(distribute, gas["Density"], gas["Masses"]/gas["Density"], binsx, binsy, binsz, grid_size)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(distr, i_group, verts) for i_group, verts in batched_tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            thread_results.append(future.result())

    derefined_mass = np.zeros((grid_size, grid_size, grid_size))
    errors = np.zeros(2)
    for result in thread_results:
        derefined_mass += result[0]
        errors += result[1]

    return derefined_mass, errors
