import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from computeNu import computeNu
from computeRandV import computeRandV
from kepler import true_to_eccentric
from eci2ecef import eci2ecef
from ecef2LonLat import ecef2LonLat
from plotting import plotOrbits3D, mercatorDisplay

## write out all necessary vars

# 1. Given orbit design parameters
T = 24 * 3600 # period (secs)
e = 0.25 # eccentricity
i = np.deg2rad(63.4) # rad, inclination
w_values_deg = [90, 270] # deg, argument of periapsis (two cases)
Omega_values_deg = [0, 45, 90, 135, 180]  # deg, longitude of ascending node 

# 2. constants
miu = 398600.0  # km^3/s^2 
O_e = 7.2921159e-5  # rad/s, Earth rotation rate
R_e = 6378.1363  # km

# 3. Mathematical constants / step size
delta_t_mins = 5
delta_t = delta_t_mins * 60  # s

# semi-major axis from the given period (T = 2*pi*sqrt(a^3/mu))
sqrt_mu_a3 = 2 * np.pi / T  # 1/s, sqrt(mu/a^3), from T = 2*pi*sqrt(a^3/mu)
a = (miu / sqrt_mu_a3 ** 2) ** (1 / 3)
p = a * (1 - e ** 2)
rp = p / (1 + e)
ra = p / (1 - e)

# spacecraft at periapsis at the epoch: t_0 = 0, nu_0 = 0; propagate one period
t_0 = 0.0
nu_0 = 0.0
t_n = t_0 + T

## set up output dir filepath + image file path
fig_dir = os.path.join('report', 'figures')
os.makedirs(fig_dir, exist_ok=True)
earth_img_path = os.path.join(fig_dir, 'equirect_map_NASA.jpg')  # NASA equirectangular Earth image

print("\nOrbit design parameters (fixed for every case)")
print(f"\tT\t= {T/3600:.4f} hr")
print(f"\te\t= {e:.4f}")
print(f"\ti\t= {np.rad2deg(i):.4f} deg")
print(f"\ta\t= {a:.4f} km")
print(f"\tp\t= {p:.4f} km")
print(f"\trp\t= {rp:.4f} km")
print(f"\tra\t= {ra:.4f} km")

## ------------------------------ TASK 1 & 2: propagate every (w, Omega) case ------------------- #

# build t_arr 'mesh' and find nu arr given orbit design params
t_arr = np.arange(t_0, t_n + delta_t/2, delta_t)
nu_arr = computeNu(t_arr, nu_0, a, e, miu)

# orbits[w_deg][Omega_deg] = {'r_eci','lon','lat'}
orbits = {w_deg: {} for w_deg in w_values_deg}

for w_deg in w_values_deg:
    w = np.deg2rad(w_deg)
    for Omega_deg in Omega_values_deg:
        Omega = np.deg2rad(Omega_deg)

        # Task 1: ECI positions and velocities throughout orbit
        r_arr, _ = computeRandV(nu_arr, a, e, Omega, i, w, miu)

        # Task 2: Use rotations to find all r in ECEF. Find lat long with given eqns
        R_ecef = eci2ecef(t_arr, r_arr, O_e)
        lon_arr, lat_arr = ecef2LonLat(R_ecef)

        orbits[w_deg][Omega_deg] = {
            'r_eci': r_arr, 'lon': lon_arr, 'lat': lat_arr,
        }

## ------------------------------ END OF TASK 1 & 2 (propagation) ------------------------------- #


## ------------------------------ TASK 1: 3D orbit plots --------------------------------------- #

# organize by omega
for w_deg in w_values_deg:
    orbits_by_Omega = {Od: orbits[w_deg][Od]['r_eci'] for Od in Omega_values_deg}

    #initialize fig
    fig, ax = plotOrbits3D(
        orbits_by_Omega, Omega_values_deg, R_e, earth_img_path,
        title=f'Orbit in ECI, $\\omega = {w_deg}\\degree$'
    )

    #save to put in report
    fig.savefig(os.path.join(fig_dir, f'orbit3d_w{w_deg}.png'), dpi=200, bbox_inches='tight')

## ------------------------------ END OF TASK 1 ------------------------------------------------ #


## ------------------------------ TASK 2: groundtrack plots ------------------------------------- #

for w_deg in w_values_deg:
    groundtracks_by_Omega = {Od: (orbits[w_deg][Od]['lon'], orbits[w_deg][Od]['lat']) for Od in Omega_values_deg} #populate dict

    #initialize fig
    fig, ax = mercatorDisplay(
        groundtracks_by_Omega, Omega_values_deg,
        title=f'Groundtrack, $\\omega = {w_deg}\\degree$', image_path=earth_img_path
    )

    #save
    fig.savefig(os.path.join(fig_dir, f'groundtrack_w{w_deg}.png'), dpi=200, bbox_inches='tight')

## ------------------------------ END OF TASK 2 -------------------------------------------------- #



## ------------------------------ TASK 3: hemisphere, crossing point, apoapsis placement --------- #

# define helpers

# t -> f(nu)
def t_of_nu(nu, ecc, sqrt_mu_a3):
    """This function finds the time since periapsis (t0=0, nu0=0) to reach current true anomaly nu (using eccentric anomaly)."""

    E = true_to_eccentric(np.rad2deg(nu), ecc)  # nu -> E (takes nu in deg)
    M = E - ecc * np.sin(E)

    #correction
    if M < 0:
        M += 2 * np.pi
    return M / sqrt_mu_a3

# phi, theta -> f(nu)
def lon_lat_at_nu(nu, w, Omega, miu):
    """This function finds the earth-relative longitude and latitude at true anomaly nu"""

    t = t_of_nu(nu, e, sqrt_mu_a3)
 

    r_arr, _ = computeRandV(np.array([nu]), a, e, Omega, i, w, miu)
    lon_arr, lat_arr = ecef2LonLat(eci2ecef(np.array([t]), r_arr, O_e))
    return lon_arr[0], lat_arr[0]


print("\n" + "=" * 70) #even spacing
print("Task 3: hemisphere, crossing point, and apoapsis placement")
print("=" * 70)

task3_results = {}

for w_deg in w_values_deg:
    w = np.deg2rad(w_deg)

    # (a-i) find majority hemisphere with phi sign change
    lat_ref = orbits[w_deg][Omega_values_deg[0]]['lat']

    #add instances that satisfy the given condition
    n_north = np.sum(lat_ref > 0)
    n_south = np.sum(lat_ref < 0)

    majority_hemisphere = 'Northern' if n_north > n_south else 'Southern'

    # (a-ii): define crossing condition (eqn 24) and find loop time + crossing latitude
    def crossing_condition(nu, w=w):
        lon1, _ = lon_lat_at_nu(nu, w, 0.0, miu)
        lon2, _ = lon_lat_at_nu(2 * np.pi - nu, w, 0.0, miu)
        return (lon1 - lon2 + np.pi) % (2 * np.pi) - np.pi

    # find crossing lat/long with existing grid, then refine with bracketing 
    nu_grid = nu_arr[(nu_arr > 0) & (nu_arr < np.pi)] # look ONLY between periapsis and apoapsis (hemisphere sign change happens once, need this for bracketing to work)
    res_grid = np.array([crossing_condition(x) for x in nu_grid]) # get actual values in prep for sign change lookup    
    sign_change_idx = np.where(np.diff(np.sign(res_grid)) != 0)[0][0] # lookup sign change idx

    # now that we have our rough find, find between time spacing with bracketing method
    nu1_cross = brentq(crossing_condition, nu_grid[sign_change_idx], nu_grid[sign_change_idx + 1], xtol=1e-12)
    nu2_cross = 2 * np.pi - nu1_cross

    # crossing times & latitute
    t1_cross = t_of_nu(nu1_cross, e, sqrt_mu_a3)
    t2_cross = t_of_nu(nu2_cross, e, sqrt_mu_a3)

    lat_cross = np.rad2deg(lon_lat_at_nu(nu1_cross, w, 0.0, miu)[1])

    # find loop time
    loop_apo = t2_cross - t1_cross 

    # (b) & (c): define eqn to get Omeg. for target lon
    lon_apo_0, lat_apo_rad = lon_lat_at_nu(np.pi, w, 0.0, miu) # get apo longitude at current w (and Omeg = 0)

    def Omega_for_target_lon(target_lon_deg, lon_apo_0=lon_apo_0):
        Omega_deg = np.rad2deg(np.deg2rad(target_lon_deg) - lon_apo_0) % 360

        # the assignment limits Omega to [0, 180] deg, so outside that use the closest allowed value
        if Omega_deg > 180:
            Omega_deg = 180.0 if Omega_deg < 270 else 0.0
        return Omega_deg

    # organize in dict with pre-placed func.
    task3_results[w_deg] = {
        'Omega_for_target_lon': Omega_for_target_lon,
    }

    print(f"\n(w = {w_deg} deg)")
    print(f"\tpoints north / south\t= {n_north} / {n_south}  (of {len(lat_ref)})")
    print(f"\tmajority hemisphere\t= {majority_hemisphere}")
    print(f"\tcrossing point nu1, nu2\t= {np.rad2deg(nu1_cross):.4f}, {np.rad2deg(nu2_cross):.4f} deg")
    print(f"\tcrossing point lat\t= {lat_cross:.4f} deg")
    print(f"\tcrossing -> apoapsis -> crossing\t= {loop_apo/3600:.4f} hr  ({majority_hemisphere} loop)")
    print(f"\tapoapsis latitude\t= {np.rad2deg(lat_apo_rad):.4f} deg  (matches majority hemisphere: {majority_hemisphere})")
    print(f"\tapoapsis longitude (Omega = 0)\t= {np.rad2deg(lon_apo_0):.4f} deg")

# part (b): 
print(f"\n(w = 90 deg) Omega to place apoapsis over:")
for place, target_lon in [('Australia (~135 deg E)', 135.0), ('South America (~-60 deg E, Brazil)', -60.0)]: #send in as tuples
    # python will go over each element in the arr, and destruct each tuple. target lon = # in second place of tuple and I pass thru pre-placed func to find Omega for target long.
    Omega_deg = task3_results[90]['Omega_for_target_lon'](target_lon)
    lon_apo = np.rad2deg(lon_lat_at_nu(np.pi, np.deg2rad(90), np.deg2rad(Omega_deg), miu)[0])
    print(f"\t{place}\t-> Omega = {Omega_deg:.4f} deg, apoapsis longitude = {lon_apo:.4f} deg")

# part (c) (same method as above):
print(f"\n(w = 270 deg) Omega to place apoapsis over:")
for place, target_lon in [('Russia (~90 deg E, central Siberia)', 90.0), ('North America (~-100 deg E)', -100.0)]:
    Omega_deg = task3_results[270]['Omega_for_target_lon'](target_lon)
    lon_apo = np.rad2deg(lon_lat_at_nu(np.pi, np.deg2rad(270), np.deg2rad(Omega_deg), miu)[0])
    print(f"\t{place}\t-> Omega = {Omega_deg:.4f} deg, apoapsis longitude = {lon_apo:.4f} deg")

## ------------------------------ END OF TASK 3 ---------------------------------------------- #

plt.show()
