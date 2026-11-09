import numpy as np
from scipy.spatial.transform import Rotation as R

def computeRandV(nu_arr, a, e, O, i, w, miu):

    # define vars
    p = a*(1 - e**2)

    # same as hw, orbit_params -> R and V is just 3-1-3 rotation 
    # can do with scipy!
    rot1 = R.from_euler('z', O)
    rot2 = R.from_euler('x', i)
    rot3 = R.from_euler('z', w)

    rotation_matrix = rot1 * rot2 * rot3

    # scaffold r and v arrays (ECI), R and V are (n+1) x 3
    arr_len = len(nu_arr)
    r_arr = np.zeros((arr_len, 3))
    v_arr = np.zeros((arr_len, 3))

    # populate arrays 
    for j in range(len(nu_arr)):

        # def current nu and r -> f(nu)
        nu = nu_arr[j]
        r = p/(1 + e*np.cos(nu))

        # find perifocal r, v (same as hw3, eqn (3)) -> f(nu)
        r_perifocal = np.array([r*np.cos(nu), r*np.sin(nu), 0])
        v_perifocal = np.sqrt(miu/p)*np.array([-np.sin(nu), (e + np.cos(nu)), 0])

        # rotate perifocal (f(nu)) -> PCI
        r_arr[j] = rotation_matrix.apply(r_perifocal)
        v_arr[j] = rotation_matrix.apply(v_perifocal)

    return r_arr, v_arr
