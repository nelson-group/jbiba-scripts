from measurments import Sim

sim1 = Sim("/u/jbiba/projects/turbulent-driving/iso-driving128-mach0.5-sol")

sim1.prop_movie("Density", norm="log", cmap="Blues_r", vmax=3.1, vmin=2.85, clabel=r"$\mathrm{log}(\rho) [M_{\odot}/\mathrm{kpc}^3]$", grid_size=2000)
# sim1.prop_movie("Velocities", norm="log", cmap="plasma_r", vmax=1.6, vmin=3.0, clabel=r"$\mathrm{log}(v)$ [km/s]", grid=500)
