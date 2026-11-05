import numpy as np

# Not permitted to use the Project 2 time-of-flight root finder for this project -- this is a direct
# Kepler-equation solver (M -> E via Newton, E -> nu), used incrementally step-by-step. (HW4 version,
# adapted so each step advances from the PREVIOUS step's E instead of re-solving from E0 every time.)

NEWTON_ITERS = 50


def eccentric_to_true(E, e):
    """E -> true anomaly nu, both in radians, wrapped to [0, 2*pi)."""
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2), np.sqrt(1 - e) * np.cos(E / 2))
    if nu < 0:
        nu += 2 * np.pi
    return nu


def solve_kepler_newton(M, e, E_guess):
    """
    Newton's method on Kepler's equation E - e*sin(E) - M = 0, starting from E_guess (the previous
    step's E, which is a good seed since M only advances by a small amount each 5-minute step).
    """
    E = E_guess
    for _ in range(NEWTON_ITERS):
        f = E - e * np.sin(E) - M
        f_prime = 1 - e * np.cos(E)
        E -= f / f_prime
    return E


def step_kepler(E_prev, e, n, dt):
    """
    Advance one fixed-size time step dt (mean motion n = 2*pi/T), incrementally: the mean anomaly
    moves by exactly n*dt, and the previous E is used as the initial guess for the new E. Returns
    the new eccentric anomaly E and true anomaly nu.
    """
    M_prev = E_prev - e * np.sin(E_prev)
    M = M_prev + n * dt

    E = solve_kepler_newton(M, e, E_guess=E_prev + n * dt)
    nu = eccentric_to_true(E, e)

    return E, nu
