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


def map_unstructured_to_structured_3d_optimized(positions, values, grid_size):
    x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
    y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
    z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])
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
    _, idx = tree.query(grid_coords, k=1)
    if len(values.shape) < 2:
        structured_grid = values[idx].reshape(grid_height, grid_width, grid_depth)
    else:
        structured_grid = values[idx].reshape(grid_height, grid_width, grid_depth, values.shape[-1])

    return structured_grid


def map_unstructured_to_structured_slice_optimized(positions, values, grid_size, grid):

    x_min, x_max = np.min(positions[:, 0]), np.max(positions[:, 0])
    y_min, y_max = np.min(positions[:, 1]), np.max(positions[:, 1])
    z_min, z_max = np.min(positions[:, 2]), np.max(positions[:, 2])

    grid_height, grid_width, grid_depth = tuple(map(len, grid))

    x_grid = np.linspace(x_min + grid[0][0]*(x_max - x_min)/grid_size, x_min + grid[0][-1]*(x_max - x_min)/grid_size, grid_width)
    y_grid = np.linspace(x_min + grid[1][0]*(y_max - y_min)/grid_size, y_min + grid[1][-1]*(y_max - y_min)/grid_size, grid_height)
    z_grid = np.linspace(x_min + grid[2][0]*(z_max - z_min)/grid_size, z_min + grid[2][-1]*(z_max - z_min)/grid_size, grid_depth)
    
    X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
    if len(values.shape) < 2:
        structured_grid = np.zeros((grid_height, grid_width, grid_depth))
    else:
        structured_grid = np.zeros((grid_height, grid_width, grid_depth, values.shape[-1]))
    tree = cKDTree(positions)
    grid_coords = np.vstack([X_grid.ravel(), Y_grid.ravel(), Z_grid.ravel()]).T
    _, idx = tree.query(grid_coords, k=1)

    if len(values.shape) < 2:
        structured_grid = values[idx].reshape(grid_height, grid_width, grid_depth)
    else:
        structured_grid = values[idx].reshape(grid_height, grid_width, grid_depth, values.shape[-1])

    if 1 in tuple(map(len, grid)):
        return np.squeeze(structured_grid)
    return structured_grid


def measure_1d_power_spectrum_vector_field(v, boxsize=1.0, bins=200):
    assert v.ndim == 4 and v.shape[-1] == 3, "Field must be 3D vector on grid"

    Nx, Ny, Nz, _ = v.shape
    Lx, Ly, Lz = (boxsize, boxsize, boxsize) if np.isscalar(boxsize) else boxsize

    v_k = np.fft.fftn(v, norm="ortho")
    power = np.sum(np.abs(v_k)**2, axis=-1)  

    kx = np.fft.fftfreq(Nx, d=Lx / Nx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny, d=Ly / Ny) * 2 * np.pi
    kz = np.fft.fftfreq(Nz, d=Lz / Nz) * 2 * np.pi
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing='ij')
    k = np.sqrt(kx**2 + ky**2 + kz**2).ravel()
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


class NoMatchingFilesError(Exception):
    """Raised when no matching files are found."""
    pass


def generate_frame(frame, property, sim, clabel=None, norm=None, grid_size=100, grid=None, cmap=None, vmin=None, vmax=None, static=False):
    fig, ax = plt.subplots(figsize=(5, 5))
    sim.prop_image(frame, property, fig, ax, cbar=True, clabel=clabel, norm=norm, grid_size=grid_size, grid=grid, cmap=cmap, vmin=vmin, vmax=vmax, static=static)
    sim.save_plot(fig, f"{property}_movie", f"{property}_{frame:03d}.png", dpi=1000)
    plt.close(fig)


class Sim:
    def __init__(self, path):
        self.path = path
        self.snaps = sorted(glob.glob(os.path.join(path, "output/snap_[0-9]*.hdf5")))
        self.time_bet_snaps = 0.00033025099*978.5 # Myr
        if not self.snaps:
            raise NoMatchingFilesError(f"No snapshots found in {path}/output/.")
        
    
    def load(self, snap, key_list=None):
        filename = self.snaps[snap]
        values = dict()
        with h5py.File(filename, 'r') as file:
            gas = file["PartType0"]
            if not key_list:
                for key in gas.keys():
                    values[key] = np.array(gas[key])
            elif isinstance(key_list, str):
                values = np.array(gas[key_list])
            else:
                for key in key_list:
                    values[key] = np.array(gas[key])
        return values

    
    def load_structured(self, snap, key_list=None, grid=100):
        values = self.load(snap, key_list)
        positions = self.load(snap, "Coordinates")
        if isinstance(values, dict):
            structured_vals = dict()
            for key in values.keys():
                structured_vals[key] = map_unstructured_to_structured_3d_optimized(positions, values[key], grid_size=(grid, grid, grid))
        else:
            structured_vals = map_unstructured_to_structured_3d_optimized(positions, values, grid_size=(grid, grid, grid))
        return structured_vals
    

    def load_structured_slice(self, snap, grid_size, grid=None, key_list=None):
        if not grid:
            # use full x and y and z slice in the middle
            grid = (range(grid_size), range(grid_size), range(grid_size//2, grid_size//2 + 1))
        values = self.load(snap, key_list)
        positions = self.load(snap, "Coordinates")
        if isinstance(values, dict):
            structured_vals = dict()
            for key in values.keys():
                structured_vals[key] = map_unstructured_to_structured_slice_optimized(positions, values[key], grid_size=grid_size, grid=grid)
        else:
            structured_vals = map_unstructured_to_structured_slice_optimized(positions, values, grid_size=grid_size, grid=grid)
        return structured_vals
    

    def load_structured_column(self):
        pass

    
    def get_fields(self):
        keys = list()
        with h5py.File(self.snaps[0], 'r') as file:
            gas = file["PartType0"]
            keys = list(gas.keys())
        return keys

    
    def save_plot(self, fig, folder, name, dpi=1000):
        plots_folder = Path(self.path + "/plots/" + folder)
        plots_folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(self.path + "/plots/" + folder + "/" + name, bbox_inches='tight', dpi=dpi)

    
    def prop_image(self, snap, key, fig=None, ax=None, cbar=True, clabel=None, grid_size=100, grid=None, save=False, norm=None, cmap=None, vmin=None, vmax=None, static=False):
        if not grid:
            # use full x and y and z slice in the middle
            grid = (range(grid_size), range(grid_size), range(grid_size//2, grid_size//2 + 1))
        prop = self.load(snap, key)
        positions = self.load(snap, "Coordinates")
        if static:
            pass
        else:
            structured_prop = map_unstructured_to_structured_slice_optimized(positions, prop, grid_size, grid=grid)
            if len(structured_prop.shape) > 3:
                structured_prop = np.sqrt(np.sum(structured_prop**2, axis=-1))
            if len(grid[2]) > 1:
                structured_prop = np.sum(structured_prop, axis=2)/len(grid[2])
        if not fig:
            fig, ax = plt.subplots(figsize=(5, 5))
        if norm == "log":
            structured_prop = np.log10(structured_prop)
        im = ax.imshow(structured_prop, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax, interpolation="none")
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax, label=clabel) 

        ax.set_xticks(np.linspace(0, structured_prop.shape[0], 7))
        ax.set_yticks(np.linspace(0, structured_prop.shape[1], 7))
        ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(0.0, 1.0, 7)])
        ax.set_yticklabels([f"{x:.1f}" for x in np.linspace(0.0, 1.0, 7)])
        ax.set_xlabel(r"$x$ [kpc]")
        ax.set_ylabel(r"$y$ [kpc]")

        if save:
            self.save_plot(fig, "general", f"{key}_slice_snap{snap}.png")
            pass
        return im, structured_prop


    def prop_movie(self, property, clabel=None, norm=None, cmap=None, vmin=None, vmax=None, grid_size=100, grid=None, threads=8, static=False):
        if not grid:
            # use full x and y and z slice in the middle
            grid = (range(grid_size), range(grid_size), range(grid_size//2, grid_size//2 + 1))
        arguments = list(range(0, len(self.snaps)))
        func = partial(generate_frame, property=property, sim=self, clabel=clabel, norm=norm, grid_size=grid_size, grid=grid, cmap=cmap, vmin=vmin, vmax=vmax, static=static)
        
        print(len(arguments))
        with Pool(processes=threads) as pool:
            results = list(tqdm(pool.imap(func, arguments), total=len(arguments)))


    def mach_number_plot(self, fig=None, ax=None, snaprange=None, save=False):
        if not snaprange:
            snaprange = range(len(self.snaps))
        data = list()
        for i in tqdm(snaprange):
            with h5py.File(self.snaps[i], 'r') as file:
                gas = file["PartType0"]
                velocities = np.array(gas["Velocities"])   
                density = np.array(gas["Density"])
                pressure = np.array(gas["Pressure"])
                acc = np.array(gas["Acceleration"])
                mass = np.array(gas["Masses"])

            volume = mass/density
            gamma = 5/3
            soundspeed = np.sqrt(gamma*pressure/density)
            c_0 = np.sum(soundspeed * volume)
            rms_vel = np.sqrt(np.sum(np.sum(velocities**2, axis=1) * volume))
            rms_acc = np.sqrt(np.sum(np.sum(acc**2, axis=1) * volume))
            mach_rms = np.sqrt(np.sum(np.sum(velocities**2, axis=1)/soundspeed**2 * volume))
            data.append([c_0, rms_vel, mach_rms, rms_acc])


        time = np.array(list(snaprange)) * self.time_bet_snaps
        data = np.array(data)
        plot1 = ax.plot(time[3:], data[3:, 1]/data[3, 0], label=r"$v_{\mathrm{rms}}/c_0$")
        plot2 = ax.plot(time[3:], data[3:, 2], label=r"$(v/c)_{\mathrm{rms}}$")
        ax.set_ylabel(r"Rms mach number $\mathcal{M}$")
        ax.set_xlabel(r"$t$ [Myr]")
        ax.legend(frameon=False)
        ax.tick_params(direction='in', top=True, right=True)
        if save:
            self.save_plot(fig, "general", f"machnumber_snap{snaprange[0]}-{snaprange[-1]}.png")
        return (plot1, plot2), data
    

    def property_pdf(self, property, snap ,fig=None, ax=None, avg=1, gaussianfit=True, save=False, hist_range=(-2.0, 1.5)):
        # if not fig:
        #     fig, ax = plt.subplots()

        mean_hist = np.zeros(1000)
        count = 0
        for j in range(avg):
            if snap + j >= len(self.snaps):
                continue
            count += 1
            with h5py.File(self.snaps[snap + j], 'r') as file:
                gas = file["PartType0"]
                density = np.array(gas["Density"])
                mass = np.array(gas["Masses"])
                prop = np.array(gas[property])
            
            if len(prop.shape) > 1:
                prop = np.sqrt(np.sum(prop**2, axis=-1))
            volume = mass/density
            prop_zero = np.sum(prop * volume)
            hist, bin_edges = np.histogram(np.log10(prop/prop_zero), bins=1000, range=hist_range, density=True, weights=volume)
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            mean_hist = mean_hist + hist

        hist = mean_hist/count
        return hist, bin_centers
        idx = hist > 1e-6
        hist = hist[idx]
        bins = bin_centers[idx]

        line = ax.plot(bins, hist)

        if gaussianfit:
            def gaussian(x, mu, sigma):
                return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-0.5 * ((x - mu)/sigma)**2)
            popt, pcov = curve_fit(gaussian, bins, hist)
            mu_fit, sigma_fit = popt
            gaussfit = ax.plot(bins, gaussian(bins, *popt), label=f'Fitted Gaussian\nμ={mu_fit:.2f}, σ={sigma_fit:.2f}', linestyle='--')
            ax.legend()
        
        ax.set_yscale("log")
        ax.set_ylim(1e-6)
        # ax.title(f"t = {0.00001321*978.5 * 200} Myr")
        # ax.set_xlabel(r'$log(\rho/\langle \rho \rangle)$')
        # plt.ylabel(r'PDF($log(\rho/\langle \rho \rangle)$)')
        if save:
            self.save_plot(fig, "general",f"{property}_pdf_snap{snap}.png")
        return line, gaussfit


    def property_pdf_evolution(self, property, fig=None, ax=None, start=0, stop=-1, num_evol=5, avg=1, save=False):
        snaprange = range(start, stop, (stop-start)//num_evol)
        times = np.array(list(range(0, stop))) * self.time_bet_snaps
        cmap = plt.get_cmap('Reds')
        norm = Normalize(vmin=times.min(), vmax=times.max())

        if not fig:
            fig, ax = plt.subplots()

        for i in snaprange:
            mean_hist = np.zeros(1000)
            count = 0
            for j in range(avg):
                if i + j >= len(self.snaps):
                    continue
                count += 1
                with h5py.File(self.snaps[i + j], 'r') as file:
                    gas = file["PartType0"]
                    density = np.array(gas["Density"])
                    mass = np.array(gas["Masses"])
                    prop = np.array(gas[property])
                
                if len(prop.shape) > 1:
                    prop = np.sqrt(np.sum(prop**2, axis=-1))
                volume = mass/density
                prop_zero = np.sum(prop * volume)
                hist, bin_edges = np.histogram(np.log10(prop/prop_zero), bins=1000, range=(-2.0, 1.5), density=True, weights=volume)
                bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                mean_hist = mean_hist + hist
                color = cmap(norm(times[i + avg//2]))

            ax.plot(bin_centers, mean_hist/count, color=color)

        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Only needed for colorbar
        plt.colorbar(sm, ax=ax, label='t [Myr]')
        ax.set_yscale("log")
        # ax.set_ylim(1e-2)
        # ax.set_xlabel(r'$log(\rho/\langle \rho \rangle)$')
        # ax.set_ylabel(r'PDF($log(\rho/\langle \rho \rangle)$)')
        fig.tight_layout()
        if save:
            self.save_plot(fig, "general",f"{property}_pdf_evolution_snap{snaprange[0]}-{snaprange[-1]}.png")


    def velocity_power_spectrum(self, snap, fig=None, ax=None, grid=100, avg=1, bins=200, save=False):
        Pk_avg = np.zeros(bins)
        count = 0
        for i in range(snap, snap + avg):
            if i >= len(self.snaps):
                continue
            count += 1
            with h5py.File(self.snaps[i], 'r') as file:
                    gas = file["PartType0"]
                    velocity = np.array(gas["Velocities"])
                    positions = np.array(gas["Coordinates"])

            structured_velocity = map_unstructured_to_structured_3d_optimized(positions, velocity, (grid, grid, grid))
            k_bins, Pk = measure_1d_power_spectrum_vector_field(structured_velocity, bins=bins)
            valid = ~np.isnan(Pk)
            Pk_avg[valid] = Pk_avg[valid] + Pk[valid]


        Pk_avg = Pk_avg/count
        nonzero = np.where(Pk_avg > 0)
        cascade = np.where((k_bins[nonzero] > 20) & (k_bins[nonzero] < 60))

        logk = np.log10(k_bins[nonzero][cascade])
        logP = np.log10(Pk_avg[nonzero][cascade])

        # Linear fit: logP = -alpha * logk + logA
        coeffs = np.polyfit(logk, logP, 1)
        slope, intercept = coeffs
        alpha = -slope
        A = 10**intercept

        plot = ax.plot(k_bins[nonzero], Pk_avg[nonzero])
        fit = ax.plot(k_bins[nonzero][cascade], A * k_bins[nonzero][cascade]**(-alpha), '--', label=f'Fit: $k^{{-{alpha:.2f}}}$')
        ax.legend()
        ax.set_xlabel("k")
        ax.set_ylabel("P(k)")
        # plt.grid(True, which="both")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.axvline(x=2*np.pi/1, color='r', linestyle='--', linewidth=1.5)
        ax.axvline(x=np.pi * grid * np.sqrt(3), color='r', linestyle='--', linewidth=1.5)
        # plt.xlabel(r"$k[kpc^2]$")
        # plt.ylabel(r"$E(k)$")
        if save:
            self.save_plot(fig, "general", f"velocity_powerspec_snap{snap}.png")
        return plot, fit, np.array([k_bins, Pk_avg])


if __name__ == "main":
    sim1 = Sim("/u/jbiba/projects/turbulent-driving/driving128-mach10-forceramp")
    data = sim1.load(20, ["Coordinates", "Velocities"])
    print(sim1.get_fields())