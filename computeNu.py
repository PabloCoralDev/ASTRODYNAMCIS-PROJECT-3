from kepler import step_kepler
import numpy as np


def computeNu(t_arr, nu_0, a, e, miu):
    """
    This function computes an array of ellapsed true anomaly values given a
    starting nu and an array of time values.

    """

    # scaffold output array (same size as t) and set IC
    nu_arr = np.zeros(len(t_arr))
    nu_arr[0] = nu_0

    # populate array starting from second position [1] and use previous values as inputs
    for i in range(1, len(t_arr)):
        nu_arr[i] = step_kepler(t_arr[i-1], nu_arr[i-1], t_arr[i], a, e, miu)

    return nu_arr
