import numpy as np

from kepler import step_kepler
from elements import elements_to_rv, MU_EARTH


def propagate_orbit(a, e, i, w, O, T, dt, mu=MU_EARTH):
    """
    Propagates one full period of the orbit starting at periapsis (t0 = 0, nu0 = 0, E0 = 0), stepping
    forward dt at a time with the incremental Kepler solver (each step's E seeds the next step -- the
    location at the end of a 5-minute interval comes from the location at the end of the previous one,
    per the assignment). Returns t_arr, nu_arr, r_arr (ECI, km), v_arr (ECI, km/s).
    """
    n = 2 * np.pi / T  # mean motion

    t_arr = np.arange(0, T + dt / 2, dt)
    n_pts = len(t_arr)

    nu_arr = np.zeros(n_pts)
    r_arr = np.zeros((n_pts, 3))
    v_arr = np.zeros((n_pts, 3))

    E = 0.0  # periapsis
    nu_arr[0] = 0.0
    r_arr[0], v_arr[0] = elements_to_rv(a, e, O, i, w, 0.0, mu=mu)

    for j in range(1, n_pts):
        E, nu = step_kepler(E, e, n, dt)
        nu_arr[j] = nu
        r_arr[j], v_arr[j] = elements_to_rv(a, e, O, i, w, nu, mu=mu)

    return t_arr, nu_arr, r_arr, v_arr
