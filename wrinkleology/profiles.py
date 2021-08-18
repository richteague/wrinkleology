"""
A series of functions to build radial profiles to describe parameter for
``simple_disk``.
"""

import numpy as np
from scipy.interpolate import interp1d


def gaussian(x, x0, dx, A):
    """Simple Gaussian function."""
    return A * np.exp(-0.5*((x - x0) / dx)**2)


def powerlaw(x, x0, q):
    """Simple powerlaw function."""
    return x0 * x**q


def tapered_powerlaw(x, x0, q, r_taper, q_taper):
    """Exponentally tapered powerlaw."""
    return x0 * x**q * np.exp(-(x / r_taper)**q_taper)


def broken_powerlaw(x, x0, q0, x1, q1, xc):
    """Broken powerlaw."""
    return np.where(x <= xc, powerlaw(x, x0, q0), powerlaw(x, x1, q1))


def make_analytical_profile(x, params, type):
    """Build the profile based on an analytical form."""
    return eval(f'{type}(x, *params)')


def make_interpolated_profile(x, x_in, y_in, y_max=None):
    """"""
    return interp1d(x_in, y_in, bounds_error=False)(x)
