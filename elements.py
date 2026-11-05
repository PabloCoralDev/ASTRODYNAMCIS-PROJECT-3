import numpy as np
from scipy.spatial.transform import Rotation as R

MU_EARTH = 398600.0  # km^3/s^2
R_EARTH = 6378.1363  # km


def elements_to_rv(a, e, O, i, w, nu, mu=MU_EARTH):
    """
    Classical orbital elements [a, e, O, i, w, nu] (angles in radians) -> ECI position and inertial velocity.
    Perifocal state rotated into the inertial frame with a 3-1-3 (Z-X-Z) rotation (O, i, w), same as Project 2.
    """
    rot1 = R.from_euler('z', O)
    rot2 = R.from_euler('x', i)
    rot3 = R.from_euler('z', w)
    rotation_matrix = rot1 * rot2 * rot3

    p = a * (1 - e ** 2)
    r = p / (1 + e * np.cos(nu))
    r_perifocal = np.array([r * np.cos(nu), r * np.sin(nu), 0.0])
    v_perifocal = np.sqrt(mu / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])

    r_eci = rotation_matrix.apply(r_perifocal)
    v_eci = rotation_matrix.apply(v_perifocal)

    return r_eci, v_eci
