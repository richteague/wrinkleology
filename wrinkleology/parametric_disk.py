"""
An example of how to use simple_disk to make a parametric disk model.

All radial profiles are described as an exponentially tapered power law profile
of the form:

y(r) = y_10 * (r / 10au)**y_q * np.exp(-(r / y_tap)**y_exp)

This means that for each of z, Tb, dV and tau there are four free parameters:
{X_10, X_q, X_tap, X_exp}. For the rotational velocity of the disk, we assume
cylindrical Keplerian rotation given by {mstar}, the stellar mass in solar
masses. We assume the disk is symmetric over the midplane, so all profiles
describe the front and back case.
"""

from .simple_disk import simple_disk
import scipy.constants as sc
import numpy as np


def tapered_powerlaw(r, y_10, y_q, y_tap, y_exp):
    """Exponentially tapered power law."""
    return y_10 * (r / 10.0)**y_q * np.exp(-(r / y_tap)**y_exp)


def parametric_disk(x0, y0, inc, PA, z_10, z_q, z_tap, z_exp, Tb_10, Tb_q,
                    Tb_tap, Tb_exp, dV_10, dV_q, dV_tap, dV_exp, tau_10, tau_q,
                    tau_tap, tau_exp, mstar, velax, vlsr, dist, FOV, npix,
                    quiet=True):
    """
    Build a parametric disk.

    Args:
        TBD

    Returns:
        TBD
    """
    # Get a simple_disk instance.

    disk = simple_disk(quiet=quiet)

    # Set up the viewing geometry, specifying the field of view (FOV) and the
    # number of pixels on each image edge (npix). This then defines the cell
    # scale for the images.

    disk.set_sky_coords(FOV=FOV, npix=npix)

    # Set up the disk geometry. We assume the front and back side of the disk
    # are symmetric.

    def z_f(r):
        """Emission surface of the front side of the disk."""
        return tapered_powerlaw(r, z_10, z_q, z_tap, z_exp)

    def z_b(r):
        """Emission surface for the back side of the disk."""
        return -z_f(r)

    disk.set_disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, dist=dist,
                         z_func=z_f, side='front')

    disk.set_disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, dist=dist,
                         z_func=z_b, side='back')

    # Set up the emission profiles. For each side of the disk the emission is
    # described by an optical depth, a line width, a peak line brightness.

    def Tb(r):
        """Peak brightness profile in [K]."""
        return tapered_powerlaw(r, Tb_10, Tb_q, Tb_tap, Tb_exp)

    def dV(r):
        """Doppler linewidth profile in [m/s]."""
        return tapered_powerlaw(r, dV_10, dV_q, dV_tap, dV_exp)

    def tau(r):
        """Optical depth profile."""
        return tapered_powerlaw(r, tau_10, tau_q, tau_tap, tau_exp)

    # For each of these values we can set limits which can be a useful way to
    # deal with the divergence of power laws close to the disk center. Although
    # this can be specified for each side independently, we assume they're the
    # same for simplicity (so setting `side='both'`).

    disk.set_Tb_profile(function=Tb, min=0.0, max=None, side='both')
    disk.set_dV_profile(function=dV, min=0.0, max=None, side='both')
    disk.set_tau_profile(function=tau, min=0.0, max=None, side='both')

    # Set up the velocity structure. Here we use a simple Keplerian rotation
    # curve, althoguh in principle anything can be used.

    def vkep(r):
        """Keplerian rotational velocity profile in [m/s]."""
        return np.sqrt(sc.G * mstar * 1.98847e30 / r / sc.au)

    disk.set_vtheta_profile(function=vkep, side='both')

    # Now we can build and return the datacube.

    return disk.get_cube(velax=velax, vlsr=vlsr)
