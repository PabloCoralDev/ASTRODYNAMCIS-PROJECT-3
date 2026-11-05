import os
import numpy as np
import matplotlib.pyplot as plt

# markers suggested in the assignment, one per Omega value
OMEGA_MARKERS = ['o', 's', 'd', 'v', '>']
OMEGA_COLORS = ['tab:red', 'tab:orange', 'tab:green', 'tab:blue', 'tab:purple']


def earthSphere(ax, R_e, image_path='earth.jpg', n=100):
    """
    Plots an earth sphere textured with an equirectangular image of earth. (Python version of
    earthSphere.m, same as Project 2.)
    """
    lon = np.linspace(-np.pi, np.pi, 2 * n)
    lat = np.linspace(np.pi / 2, -np.pi / 2, n)
    lon, lat = np.meshgrid(lon, lat)

    x = R_e * np.cos(lat) * np.cos(lon)
    y = R_e * np.cos(lat) * np.sin(lon)
    z = R_e * np.sin(lat)

    if os.path.exists(image_path):
        img = plt.imread(image_path)
        if img.dtype == np.uint8:
            img = img / 255

        rows = np.linspace(0, img.shape[0] - 1, n).astype(int)
        cols = np.linspace(0, img.shape[1] - 1, 2 * n).astype(int)
        colors = img[np.ix_(rows, cols)][..., :3]

        ax.plot_surface(x, y, z, facecolors=colors, rstride=1, cstride=1, linewidth=0, shade=False)
    else:
        ax.plot_surface(x, y, z, color='tab:blue', alpha=0.6, linewidth=0)


def plotOrbits3D(orbits_by_Omega, Omega_values, R_e, image_path, title, view_az=49.5, view_el=22.8):
    """
    One 3D figure with the earth and one orbit trace per Omega value, color-coded and legended.

    orbits_by_Omega -> dict {Omega_deg: r_arr (ECI, (n,3))}
    """
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection='3d')

    earthSphere(ax, R_e, image_path=image_path)

    for idx, Omega_deg in enumerate(Omega_values):
        r_arr = orbits_by_Omega[Omega_deg]
        ax.plot(r_arr[:, 0], r_arr[:, 1], r_arr[:, 2], linewidth=2,
                color=OMEGA_COLORS[idx], label=f'$\\Omega = {Omega_deg}\\degree$')

    ax.set_aspect('equal')
    ax.view_init(elev=view_el, azim=view_az - 90)
    ax.set_xlabel('x (km)'); ax.set_ylabel('y (km)'); ax.set_zlabel('z (km)')
    ax.set_title(title)
    ax.legend(loc='upper left')

    return fig, ax


def mercatorDisplay(groundtracks_by_Omega, Omega_values, title, image_path='earth.jpg'):
    """
    Groundtrack figure on an equirectangular Earth image, one marker style per Omega value.

    groundtracks_by_Omega -> dict {Omega_deg: (lon_arr_rad, lat_arr_rad)}
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    if os.path.exists(image_path):
        ax.imshow(plt.imread(image_path), extent=[-180, 180, -90, 90])

    for idx, Omega_deg in enumerate(Omega_values):
        lon_arr, lat_arr = groundtracks_by_Omega[Omega_deg]
        ax.plot(np.rad2deg(lon_arr), np.rad2deg(lat_arr), OMEGA_MARKERS[idx], markersize=4,
                color=OMEGA_COLORS[idx], label=f'$\\Omega = {Omega_deg}\\degree$')

    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_box_aspect(1 / 2)
    ax.set_xlabel('Earth-relative longitude (deg)')
    ax.set_ylabel('Geocentric latitude (deg)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8, ncol=5)

    return fig, ax
