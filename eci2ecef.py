import numpy as np
from dcmI2E import dcmI2E

def eci2ecef(t_arr, r_arr, O_e):
    """
    This function transforms every row of incoming r_arr (in ECI) itno ECEF with it's matching time in
    t_arr, and returns an array of positions transformed from ECI to ECEF. The inputs are: 

    O_e -> angular rate in radians per TU
    t_arr -> arr of times, units MUST match TU in O_e
    r_ arr -> arr of radii

    """
    # scaffold new arr (r size (n+1)x3)
    R_ecef = np.zeros((len(t_arr), 3))

    # dynamically calculate rotation matrix for every t-value corresp. to r.
    for j in range(len(t_arr)):
        R_ecef[j] = dcmI2E(t_arr[j], O_e) @ r_arr[j] # @ operator does the same as .apply()

    return R_ecef
