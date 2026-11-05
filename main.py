import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from elements import MU_EARTH, R_EARTH
from propagate_orbit import propagate_orbit
from eci2ecef import eci2ecef
from ecef2LonLat import ecef2LonLat
from plotting import plotOrbits3D, mercatorDisplay

## write out all necessary vars

# 1. Given orbit design parameters
T = 24 * 3600          # s, orbital period
e = 0.25                # eccentricity
i = np.deg2rad(63.4)    # rad, inclination
w_values_deg = [90, 270]        # deg, argument of periapsis (two cases)
Omega_values_deg = [0, 45, 90, 135, 180]  # deg, longitude of ascending node sweep

# 2. Physical constants
O_e = 7.2921159e-5  # rad/s, Earth rotation rate

# 3. Mathematical constants / step size
delta_t_mins = 5
delta_t = delta_t_mins * 60  # s

# semi-major axis from the given period (T = 2*pi*sqrt(a^3/mu))
n_mean_motion = 2 * np.pi / T
a = (MU_EARTH / n_mean_motion ** 2) ** (1 / 3)
p = a * (1 - e ** 2)
rp = p / (1 + e)
ra = p / (1 - e)

## set up output dir filepath + image file path
fig_dir = os.path.join('report', 'figures')
os.makedirs(fig_dir, exist_ok=True)
earth_img_path = os.path.join(fig_dir, 'equirect_map_NASA.jpg')  # NASA equirectangular Earth image (same as Project 2)

print("\nOrbit design parameters (fixed for every case)")
print(f"\tT\t= {T/3600:.4f} hr")
print(f"\te\t= {e:.4f}")
print(f"\ti\t= {np.rad2deg(i):.4f} deg")
print(f"\ta\t= {a:.4f} km")
print(f"\tp\t= {p:.4f} km")
print(f"\trp\t= {rp:.4f} km")
print(f"\tra\t= {ra:.4f} km")

## ------------------------------ TASK 1 & 2: propagate every (w, Omega) case ---------------- #

# orbits[w_deg][Omega_deg] = {'t','nu','r_eci','v_eci','lon','lat'}
orbits = {w_deg: {} for w_deg in w_values_deg}

for w_deg in w_values_deg:
    w = np.deg2rad(w_deg)
    for Omega_deg in Omega_values_deg:
        Omega = np.deg2rad(Omega_deg)

        # Task 1: incremental Kepler-solver propagation, one full period, 5-minute steps
        t_arr, nu_arr, r_arr, v_arr = propagate_orbit(a, e, i, w, Omega, T, delta_t)

        # Task 2: ECI -> ECEF -> Earth-relative longitude / geocentric latitude
        R_ecef = eci2ecef(t_arr, r_arr, O_e)
        lon_arr, lat_arr = ecef2LonLat(R_ecef)

        orbits[w_deg][Omega_deg] = {
            't': t_arr, 'nu': nu_arr, 'r_eci': r_arr, 'v_eci': v_arr,
            'lon': lon_arr, 'lat': lat_arr,
        }

## ------------------------------ END OF TASK 1 & 2 (propagation) ----------------------------- #


## ------------------------------ TASK 1: 3D orbit plots --------------------------------------- #

for w_deg in w_values_deg:
    orbits_by_Omega = {Od: orbits[w_deg][Od]['r_eci'] for Od in Omega_values_deg}
    fig, ax = plotOrbits3D(
        orbits_by_Omega, Omega_values_deg, R_EARTH, earth_img_path,
        title=f'Orbit in ECI, $\\omega = {w_deg}\\degree$'
    )
    fig.savefig(os.path.join(fig_dir, f'orbit3d_w{w_deg}.png'), dpi=200, bbox_inches='tight')

## ------------------------------ END OF TASK 1 ------------------------------------------------ #


## ------------------------------ TASK 2: groundtrack plots ------------------------------------- #

for w_deg in w_values_deg:
    groundtracks_by_Omega = {Od: (orbits[w_deg][Od]['lon'], orbits[w_deg][Od]['lat']) for Od in Omega_values_deg}
    fig, ax = mercatorDisplay(
        groundtracks_by_Omega, Omega_values_deg,
        title=f'Groundtrack, $\\omega = {w_deg}\\degree$', image_path=earth_img_path
    )
    fig.savefig(os.path.join(fig_dir, f'groundtrack_w{w_deg}.png'), dpi=200, bbox_inches='tight')

## ------------------------------ END OF TASK 2 -------------------------------------------------- #


## ------------------------------ TASK 3: hemisphere, crossing point, apoapsis placement --------- #

def nu_to_E(nu, ecc):
    E = 2 * np.arctan2(np.sqrt(1 - ecc) * np.sin(nu / 2), np.sqrt(1 + ecc) * np.cos(nu / 2))
    if E < 0:
        E += 2 * np.pi
    return E


def t_of_nu(nu, ecc, n_mm):
    """Time since periapsis (t0=0, nu0=0) to reach true anomaly nu, via Kepler's equation (no root-finding needed: direct E -> M)."""
    E = nu_to_E(nu, ecc)
    M = E - ecc * np.sin(E)
    if M < 0:
        M += 2 * np.pi
    return M / n_mm


def lon_orbital(u, incl):
    """Earth-relative longitude swept along the orbital plane (measured from the ascending node), via the spherical right triangle at inclination incl."""
    return np.arctan2(np.cos(incl) * np.sin(u), np.cos(u))


t_apo = T / 2  # time from periapsis to apoapsis: nu=pi -> E=pi -> M=pi -> t = T/2 exactly

print("\n" + "=" * 70)
print("  Task 3: hemisphere dwell, crossing point, and apoapsis placement")
print("=" * 70)

task3_results = {}

for w_deg in w_values_deg:
    w = np.deg2rad(w_deg)

    # part (a-i): majority hemisphere, using any one Omega (hemisphere structure is Omega-invariant)
    lat_ref = orbits[w_deg][Omega_values_deg[0]]['lat']
    frac_north = np.mean(lat_ref > 0)
    majority_hemisphere = 'Northern' if frac_north > 0.5 else 'Southern'

    # part (a-ii): crossing point -- the point visited twice per period (ascending & descending legs share lat/lon).
    # By symmetry, lat(nu) = lat(pi - w - nu + w)... solved directly as a root-find on the longitude match
    # between a point at argument of latitude u1 and its latitude-mirror u2 = pi - u1.
    def residual(nu1, w=w):
        u1 = w + nu1
        u2 = np.pi - u1
        nu2 = (u2 - w) % (2 * np.pi)
        t1 = t_of_nu(nu1, e, n_mean_motion)
        t2 = t_of_nu(nu2, e, n_mean_motion)
        lon1 = lon_orbital(u1, i) - O_e * t1
        lon2 = lon_orbital(u2, i) - O_e * t2
        return (lon1 - lon2 + np.pi) % (2 * np.pi) - np.pi

    # bracket found by scanning nu in (0, pi); true anomaly range where the ascending leg lives
    nu_scan = np.linspace(1e-3, np.pi - 1e-3, 2000)
    res_scan = np.array([residual(x) for x in nu_scan])
    sign_change_idx = np.where(np.diff(np.sign(res_scan)) != 0)[0][0]
    nu1_cross = brentq(residual, nu_scan[sign_change_idx], nu_scan[sign_change_idx + 1], xtol=1e-12)

    u1_cross = w + nu1_cross
    u2_cross = np.pi - u1_cross
    nu2_cross = (u2_cross - w) % (2 * np.pi)
    t1_cross = t_of_nu(nu1_cross, e, n_mean_motion)
    t2_cross = t_of_nu(nu2_cross, e, n_mean_motion)
    lat_cross = np.rad2deg(np.arcsin(np.sin(i) * np.sin(u1_cross)))

    leg_short = t2_cross - t1_cross       # crossing point -> crossing point, short way
    leg_long = T - leg_short              # crossing point -> crossing point, long way (completes the period)

    # part (b)/(c): apoapsis longitude is a rigid function of Omega -> apoapsis_lon = Omega + lon_orbital(u_apo) - O_e*t_apo
    u_apo = w + np.pi
    lon_orb_apo = lon_orbital(u_apo, i)
    lat_apo = np.rad2deg(np.arcsin(np.sin(i) * np.sin(u_apo)))

    def Omega_for_target_lon(target_lon_deg, lon_orb_apo=lon_orb_apo):
        Omega_rad = np.deg2rad(target_lon_deg) - lon_orb_apo + O_e * t_apo
        return np.rad2deg(Omega_rad) % 360

    task3_results[w_deg] = {
        'majority_hemisphere': majority_hemisphere, 'frac_north': frac_north,
        'lat_cross': lat_cross, 'leg_short_hr': leg_short / 3600, 'leg_long_hr': leg_long / 3600,
        'lat_apo': lat_apo, 'Omega_for_target_lon': Omega_for_target_lon,
    }

    print(f"\n(w = {w_deg} deg)")
    print(f"\tmajority hemisphere\t= {majority_hemisphere} ({100*frac_north if majority_hemisphere=='Northern' else 100*(1-frac_north):.2f}% of orbit)")
    print(f"\tcrossing point lat\t= {lat_cross:.4f} deg")
    print(f"\tdwell time (short leg)\t= {leg_short/3600:.4f} hr")
    print(f"\tdwell time (long leg)\t= {leg_long/3600:.4f} hr")
    print(f"\tapoapsis latitude\t= {lat_apo:.4f} deg  (matches majority hemisphere: {majority_hemisphere})")

# part (b): Omega placing apoapsis over Australia and South America, w = 90 deg
print(f"\n(w = 90 deg) Omega placing apoapsis over named locations:")
for place, target_lon in [('Australia (~135 deg E)', 135.0), ('South America (~-60 deg E, Brazil)', -60.0)]:
    Omega_deg = task3_results[90]['Omega_for_target_lon'](target_lon)
    print(f"\t{place}\t-> Omega = {Omega_deg:.4f} deg")

# part (c): Omega placing apoapsis over Russia and North America, w = 270 deg
print(f"\n(w = 270 deg) Omega placing apoapsis over named locations:")
for place, target_lon in [('Russia (~90 deg E, central Siberia)', 90.0), ('North America (~-100 deg E)', -100.0)]:
    Omega_deg = task3_results[270]['Omega_for_target_lon'](target_lon)
    print(f"\t{place}\t-> Omega = {Omega_deg:.4f} deg")

## ------------------------------ END OF TASK 3 ---------------------------------------------- #

plt.show()
