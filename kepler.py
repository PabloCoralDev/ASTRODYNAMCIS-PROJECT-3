import numpy as np

# Kepler solver (nu0 -> E0 -> E(t) -> nu(t)) and a per-step wrapper, step_kepler.


# necessary eqns and functions
def true_to_eccentric(nu_deg, e):
    """Eq. 3.23 in notes"""
    nu = np.radians(nu_deg)
    E = 2 * np.arctan2(np.sqrt(1 - e) * np.sin(nu / 2), np.sqrt(1 + e) * np.cos(nu / 2))
    if E < 0:
        E += 2 * np.pi

    return E


def eccentric_to_true(E, e):
    """Eq. 3.21 in notes"""
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    if nu < 0:
        nu += 2 * np.pi

    return np.degrees(nu)


def solve_kepler(E0, e, a, t0, t, mu):
    """
    Solve Kepler's equation E - e*sin(E) = C for E at time t, given E0 at t0,
    using fpi, starting with the mean anomaly guess M0, and running for N iters.
    """
    n = np.sqrt(mu / a ** 3)
    k = int(np.floor((n * (t - t0) + (E0 - e * np.sin(E0))) / (2 * np.pi)))  # calculate int # of crossings; solve 3.48 for k.

    C = n * (t - t0) - 2 * np.pi * k + (E0 - e * np.sin(E0))  # Eq. (3.52)

    M0 = E0 - e * np.sin(E0) # mean anomaly at t0
    E = M0  # set initial guess to M0 (3.57)

    N = int(10 * np.ceil(1 / (1 - e)))  # proper num of iters
    for _ in range(N):
        E = e * np.sin(E) + C  # Eq. (3.56): E(k+1) = e*sin(E(k)) + C

    return E


def step_kepler(t_prev, nu_prev, t, a, e, mu):
    """
    One step of the Kepler solver: the end of the previous step (t_prev, nu_prev) is the start of
    this one. nu in rad (in and out).
    """
    E0 = true_to_eccentric(np.degrees(nu_prev), e)
    E = solve_kepler(E0, e, a, t_prev, t, mu=mu)
    nu = np.radians(eccentric_to_true(E, e))

    return nu
