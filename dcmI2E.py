import numpy as np

def dcmI2E(t, O_e):
    """
    This function computes the transformation matrix T_I2E that performs a rotation from ECI coordinates to ECEF via
    the earth's rate of rotation O_e and the time at which the transformation is desired, t.
    """
    # rename for simplicity
    eta = O_e*t

    # fill out direction cosine matrix
    T_I2E = np.array([
        [ np.cos(eta), np.sin(eta), 0],
        [-np.sin(eta), np.cos(eta), 0],
        [0, 0, 1]
    ])

    return T_I2E

