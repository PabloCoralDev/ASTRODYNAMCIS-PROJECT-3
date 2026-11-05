import numpy as np

def ecef2LonLat(R_ecef):
    """
    This function takes in the R_ecef matrix found in 3-2, and calculates/returns the earth-relative latitude
    and longitude arrays.
    """

    # organize components into their own arrays
    x = R_ecef[:, 0]
    y = R_ecef[:, 1]
    z = R_ecef[:, 2]
    
    # numpy can do element-per-element ops. No need for loops!
    lon_arr = np.arctan2(y, x)
    lat_arr = np.arctan2(z, np.sqrt(x**2 + y**2))

    return lon_arr, lat_arr
