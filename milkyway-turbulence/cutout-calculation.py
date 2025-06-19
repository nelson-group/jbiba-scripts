import numpy as np
import nicegalaxy
from tqdm import tqdm
import h5py
from numpy.fft import fftn, ifftn, fftfreq
from scipy.stats import binned_statistic
import concurrent.futures


measurments = [
    "subhaloId", "distGC", "distBH","SFR", 
    "DensityMean_v", "DensityDisp_v", "DensityMean_m", "DensityDisp_m",
    "LogDensityDisp_v", "LogDensityDisp_m",
    "VelocityMean_v", "VeloctiyDisp_v", "VelocityMean_m", "VelocityDisp_m",
    "Mach_v", "Mach_m", "SoundspeedMean_v", "SoundspeedMean_m",
    "PressureMean_v", "PressureDisp _v", "PressureMean_m", "PressureDisp_m",
    "LogPressureDisp_m", "LogPressureDisp_v",
    "MagneticFieldMean_m", "MagneticFieldMean_v", 
    "MagneticPressMean_v", "MagneticPressMean_m", "betaMean_v", "betaMean_m",
    "TemperatureMean_v", "MeanElectronAbundance_v", "MeanMolecularWeightMean_v",
    "TemperatureMean_m", "MeanElectronAbundance_m", "MeanMolecularWeightMean_m",
    "galactic_cos", "CompressiveFraction", "SolenoidalFraction", "CompressiveMach_v",
    "LogDensityDispRadial_v", "LogDensityDispRadial_m",
    "LogPressureDispRadial_v", "LogPressureDispRadial_m",
    "Mach_v_turb", "Mach_m_turb", "CompressiveMach_v_turb", "CompressiveFraction_turb", "SolenoidalFraction_turb"
]


def mean(vals, weights):
    if len(vals.shape) == 1:
        return np.sum(vals*weights)/np.sum(weights)
    else:
        return np.sum(vals*weights[:, None], axis=0)/np.sum(weights)


def variance(vals, weights):
    return np.sqrt(mean(vals**2, weights) - mean(vals, weights)**2)


def rms(vals, weights):
    return np.sqrt(np.sum(np.sum(vals**2, axis=-1)*weights)/np.sum(weights))


def helmholtz_decomposition(vec_field):
    N = vec_field.shape[0]
    V_hat = np.stack([fftn(vec_field[..., i]) for i in range(3)], axis=-1)
    k = fftfreq(N) * N
    kx, ky, kz = np.meshgrid(k, k, k, indexing='ij')
    k_squared = kx**2 + ky**2 + kz**2
    k_squared[k_squared == 0] = 1
    k_vec = np.stack((kx, ky, kz), axis=-1)
    dot = np.sum(V_hat * k_vec, axis=-1, keepdims=True)
    compressive_hat = dot * k_vec / k_squared[..., None]
    solenoidal_hat = V_hat - compressive_hat
    compressive = np.stack([np.real(ifftn(compressive_hat[..., i])) for i in range(3)], axis=-1)    
    solenoidal = np.stack([np.real(ifftn(solenoidal_hat[..., i])) for i in range(3)], axis=-1)
    return compressive, solenoidal



def cutout_measurments(positions, velocities, density, dens_copy, internal_energy, masses, magneticfield, electronabundance, normal_vector, center, bh_coordinates, press_copy, turb_velocities):
    vols = masses/density
    soundspeed = np.sqrt(5/3 * (5/3 - 1) * internal_energy)
    pressure = (5/3 - 1)*density*internal_energy
    molecular_weight = 4/(1 + 3*0.76 + 4*0.76*electronabundance) * (1.6726e-24)
    temperature = (5/3 - 1) * internal_energy/(1.3807e-16) * (1e10) * molecular_weight # 1e10 not 10e10

    # density stuff
    dens_mean_v = mean(density, vols)
    dens_mean_m = mean(density, masses)

    s_v = np.log(density/dens_mean_v) # log not log10
    s_m = np.log(density/dens_mean_m)
    
    # machnumber stuff
    vel_bulk_v = mean(velocities, vols)
    vel_rms_v = rms(velocities - vel_bulk_v, vols)
    vel_bulk_m = mean(velocities, masses)
    vel_rms_m = rms(velocities - vel_bulk_m, masses)

    # pressure stuff
    press_mean_v = mean(pressure, vols)
    press_mean_m = mean(pressure, masses)
    ps_v = np.log(pressure/press_mean_v)
    ps_m = np.log(pressure/press_mean_m)

    # magentic stuff
    press_mag_v = mean(np.sum(magneticfield**2, axis=-1), vols)/(8*np.pi)
    press_mag_m = mean(np.sum(magneticfield**2, axis=-1), masses)/(8*np.pi)

    # new stuff
    costheta = np.dot(normal_vector, center - bh_coordinates)/np.linalg.norm(center - bh_coordinates)

    structured_vels = nicegalaxy.map_unstructured_to_structured_3d_normal(positions, velocities - vel_bulk_v, grid_size=(100, 100, 100))
    comp, sol = helmholtz_decomposition(structured_vels)
    total_norm = np.linalg.norm(structured_vels)
    compressive_norm = np.linalg.norm(comp)
    solenoidal_norm = np.linalg.norm(sol)
    fraction_compressive = compressive_norm / total_norm
    fraction_solenoidal = solenoidal_norm/total_norm
    structured_soundspeed = nicegalaxy.map_unstructured_to_structured_3d_normal(positions, soundspeed, grid_size=(100, 100, 100))
    comp_mach = np.sqrt(np.mean(np.sum(comp**2, axis=-1)/structured_soundspeed**2))

    s_corrected = np.log(density/dens_copy)
    ln_p_corrected = np.log(pressure/press_copy)

    structured_vels_turb = nicegalaxy.map_unstructured_to_structured_3d_normal(positions, turb_velocities, grid_size=(100, 100, 100))
    comp_turb, sol_turb = helmholtz_decomposition(structured_vels_turb)
    total_norm_turb = np.linalg.norm(structured_vels_turb)
    compressive_norm_turb = np.linalg.norm(comp_turb)
    solenoidal_norm_turb = np.linalg.norm(sol_turb)
    fraction_compressive_turb = compressive_norm_turb / total_norm_turb
    fraction_solenoidal_turb = solenoidal_norm_turb/total_norm_turb
    comp_mach_turb = np.sqrt(np.mean(np.sum(comp_turb**2, axis=-1)/structured_soundspeed**2))


    return [
        0, 0.0, 0.0, 0.0,
        dens_mean_v, variance(density, vols), dens_mean_m, variance(density, masses),
        variance(s_v, vols), variance(s_m, masses),
        np.linalg.norm(vel_bulk_v), vel_rms_v, np.linalg.norm(vel_bulk_m), vel_rms_m,
        rms((velocities - vel_bulk_v)/soundspeed[:, np.newaxis], vols), rms((velocities - vel_bulk_m)/soundspeed[:, np.newaxis], masses), mean(soundspeed, vols), mean(soundspeed, masses),
        press_mean_v, variance(pressure, vols), press_mean_m, variance(pressure, masses),
        variance(ps_v, vols), variance(ps_m, masses),
        np.linalg.norm(mean(magneticfield, vols)), np.linalg.norm(mean(magneticfield, masses)),
        press_mag_v, press_mag_m, press_mean_v/press_mag_v, press_mean_m/press_mag_m,
        mean(temperature, vols), mean(electronabundance, vols), mean(molecular_weight, vols),
        mean(temperature, masses), mean(electronabundance, masses), mean(molecular_weight, masses),
        costheta, fraction_compressive, fraction_solenoidal, comp_mach,
        variance(s_corrected, vols), variance(s_corrected, masses),
        variance(ln_p_corrected, vols), variance(ln_p_corrected, masses),
        rms(turb_velocities/soundspeed[:, np.newaxis], vols), rms(turb_velocities/soundspeed[:, np.newaxis], masses), comp_mach_turb, fraction_compressive_turb, fraction_solenoidal_turb
    ]


def galaxy_scan(smoothing=3, boxes=20):
    table = {key: np.zeros(50 * boxes**3, dtype=np.float64) for key in measurments}

    for galaxy_idx in tqdm(range(50), desc="Looping over Galaxies"):
        galaxy = nicegalaxy.Galaxy(nicegalaxy.galaxies[galaxy_idx])
        with h5py.File(f"data/turb-vels/{nicegalaxy.galaxies[galaxy_idx]}.h5", "r") as f:
            turb = dict()
            for key in f:
                turb[key] = np.array(f[key])
        
        box_length = galaxy.gas["Coordinates"][:, 0].max() - galaxy.gas["Coordinates"][:, 0].min()

        min_corner = galaxy.gas["Coordinates"].min(axis=0)
        max_corner = galaxy.gas["Coordinates"].max(axis=0)
        cutout_size = box_length/2**smoothing

        spacing = (max_corner - min_corner - cutout_size) / (boxes - 1)

        distances = np.linalg.norm(galaxy.gas["Coordinates"]- galaxy.bhs["Coordinates"][0], axis=-1)
        bin_edges = np.logspace(np.log10(distances.min()), np.log10(distances.max()), 500)
        vols = galaxy.gas["Masses"]/galaxy.gas["Density"]
        hist1, _, _ = binned_statistic(distances, galaxy.gas["Density"] * vols, bins=bin_edges, statistic='sum')
        hist2, _, _ = binned_statistic(distances, vols, bins=bin_edges, statistic='sum')
        with np.errstate(divide='ignore', invalid='ignore'):
            hist = np.where((~np.isnan(hist2)) & (~np.isnan(hist1)) & (hist2 != 0), hist1 / hist2, 0.0)
        dist_bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

        zero_indices = np.where(hist == 0)[0]
        nonzero_indices = np.where(hist != 0)[0]
        for i in zero_indices:
            closest_j = nonzero_indices[np.argmin(np.abs(dist_bin_centers[nonzero_indices] - dist_bin_centers[i]))]
            hist[i] = hist[closest_j]
        distances = np.linalg.norm(galaxy.gas["Coordinates"]- galaxy.bhs["Coordinates"][0], axis=-1)
        idx = np.abs(distances[:, None] - dist_bin_centers[~np.isnan(hist)& (hist != 0)][None, :]).argmin(axis=1)
        dens_copy = hist[idx]

        hist1, _, _ = binned_statistic(distances, (5/3 - 1) * galaxy.gas["Density"] * galaxy.gas["InternalEnergy"] * vols, bins=bin_edges, statistic='sum')
        hist2, _, _ = binned_statistic(distances, vols, bins=bin_edges, statistic='sum')
        with np.errstate(divide='ignore', invalid='ignore'):
            hist = np.where((~np.isnan(hist2)) & (~np.isnan(hist1)) & (hist2 != 0), hist1 / hist2, 0.0)
        dist_bin_centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])

        zero_indices = np.where(hist == 0)[0]
        nonzero_indices = np.where(hist != 0)[0]
        for i in zero_indices:
            closest_j = nonzero_indices[np.argmin(np.abs(dist_bin_centers[nonzero_indices] - dist_bin_centers[i]))]
            hist[i] = hist[closest_j]
        distances = np.linalg.norm(galaxy.gas["Coordinates"]- galaxy.bhs["Coordinates"][0], axis=-1)
        idx = np.abs(distances[:, None] - dist_bin_centers[~np.isnan(hist)& (hist != 0)][None, :]).argmin(axis=1)
        press_copy = hist[idx]


        def cutout_calculation(i, j, k):
            cutout_min = min_corner + np.array([i, j, k]) * spacing
            cutout_max = cutout_min + cutout_size
            in_box = (
                (galaxy.gas["Coordinates"][:, 0] >= cutout_min[0]) & (galaxy.gas["Coordinates"][:, 0] < cutout_max[0]) &
                (galaxy.gas["Coordinates"][:, 1] >= cutout_min[1]) & (galaxy.gas["Coordinates"][:, 1] < cutout_max[1]) &
                (galaxy.gas["Coordinates"][:, 2] >= cutout_min[2]) & (galaxy.gas["Coordinates"][:, 2] < cutout_max[2])
            )
            if len(galaxy.gas["Coordinates"][in_box]) == 0:
                return
            center = 0.5 * (cutout_max + cutout_min)
            closest_bh_dist = np.min(np.linalg.norm(galaxy.bhs["Coordinates"] - center, axis=-1))
            gc_dist = np.linalg.norm(np.linalg.norm(galaxy.bhs["Coordinates"][0] - center))
            measurment = cutout_measurments(galaxy.gas["Coordinates"][in_box], galaxy.gas["Velocities"][in_box], galaxy.gas["Density"][in_box], dens_copy[in_box], galaxy.gas["InternalEnergy"][in_box],
                                            galaxy.gas["Masses"][in_box], galaxy.gas["MagneticField"][in_box], galaxy.gas["ElectronAbundance"][in_box], galaxy.normal_vector, center, galaxy.bhs["Coordinates"][0], press_copy[in_box], turb["TurbVelocities"][in_box])
            measurment[0] = nicegalaxy.galaxies[galaxy_idx]
            measurment[1] = gc_dist
            measurment[2] = closest_bh_dist
            measurment[3] = np.sum(galaxy.gas["StarFormationRate"][in_box])

            cutout_idx = i * boxes**2 + j * boxes + k
            for mes_idx, name in enumerate(measurments):
                table[name][galaxy_idx * boxes**3 + cutout_idx] = measurment[mes_idx]

        tasks = [(i, j, k) for i in range(boxes) for j in range(boxes) for k in range(boxes)]
        thread_results = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(cutout_calculation, i, j, k) for i, j, k in tasks]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                thread_results.append(future.result())

    return table


table_coarse = galaxy_scan(smoothing=3, boxes=10)
with h5py.File('data/new_turbulence60kpc_withturb_right.h5', 'w') as f:
    for key, array in table_coarse.items():
        f.create_dataset(key, data=array)

# table_fine = galaxy_scan(smoothing=4, boxes=10)
# with h5py.File('data/new_turbulence30kpc_withturb_right.h5', 'w') as f:
#     for key, array in table_fine.items():
#         f.create_dataset(key, data=array)

# table_finer = galaxy_scan(smoothing=5, boxes=10)
# with h5py.File('data/new_turbulence15kpc_withturb_right.h5', 'w') as f:
#     for key, array in table_finer.items():
#         f.create_dataset(key, data=array)