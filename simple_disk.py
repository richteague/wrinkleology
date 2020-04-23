import numpy as np
import scipy.constants as sc
from scipy.special import erf
import matplotlib.pyplot as plt
from astropy.convolution import convolve, Gaussian2DKernel


class simple_disk:
    """
    Simple disk structure to explore analytic kinematic features.

    Args:
        inc (float): Inclination of the source in [degrees].
        PA (float): Position angle of the source in [degrees].
        x0 (Optional[float]): Source center offset along x-axis in [arcsec].
        y0 (Optional[float]): Source center offset along y-axis in [arcsec].
        dist (Optional[float]): Distance to the source in [pc].
        mstar (Optional[float]): Mass of the central star in [Msun].
        FOV (Optional[float]): Field of view of the model in [arcsec].
        Npix (Optional[int]): Number of pixels along each axis.
        Tb0 (Optional[float]): Brightness temperature in [K] at 1 arcsec.
        Tbq (Optional[float]): Exponent of the brightness temperature
            radial profile.
        dV0 (Optional[float]): Doppler line width in [m/s] at 1 arcsec.
        dVq (Optional[float]): Exponent of the line width radial profile.
    """

    mu = 2.37
    fwhm = 2.*np.sqrt(2.*np.log(2.))
    msun = 1.988e30
    nwrap = 3

    def __init__(self, inc, PA, x0=0.0, y0=0.0, dist=100.0, mstar=1.0, FOV=3.0,
                 Npix=128, Tb0=50.0, Tbq=-1.0, Tbmax=100, dV0=200.0, dVq=-0.6,
                 dVmax=300.0, r_max=None, r_min=None):
        self.x0 = x0
        self.y0 = y0
        self.inc = inc
        self.PA = PA
        self.dist = dist
        self.mstar = mstar
        self.r_min = 0.0 if r_min is None else r_min
        self.r_max = dist * FOV / 2.0 if r_max is None else r_max
        self.set_FOV(FOV=FOV, Npix=Npix)
        self.set_brightness(Tb0=Tb0, Tbq=Tbq, Tbmax=Tbmax)
        self.set_linewidth(dV0=dV0, dVq=dVq, dVmax=dVmax)

    def set_FOV(self, FOV, Npix):
        """
        Populates the on-sky pixels and deprojected pixels.
        """

        # Save the variables.

        self.FOV = FOV
        self.Npix = int(Npix)

        # Define the sky coordinates in [arcsec].

        self.x_sky = np.linspace(-FOV/2.0, FOV/2.0, Npix)
        self.y_sky = np.linspace(-FOV/2.0, FOV/2.0, Npix)
        self.cell_sky = np.diff(self.y_sky).mean()
        self.x_sky, self.y_sky = np.meshgrid(self.x_sky, self.y_sky)

        # Define the disk coordinates in [au].

        self.x_disk = self.x_sky * self.dist
        self.y_disk = self.y_sky * self.dist
        self.r_disk = np.hypot(self.y_disk, self.x_disk)
        self.t_disk = np.arctan2(self.y_disk, self.x_disk)
        self.cell_disk = np.diff(self.y_disk).mean()

        # Calculate the projected pixel coordinates in [arcsec].

        c = self.disk_coords(x0=self.x0, y0=self.y0, inc=self.inc, PA=self.PA)
        self.r_sky = c[0]
        self.t_sky = c[1]
        self.z_sky = c[2]

    def in_disk(self, projection='sky'):
        """
        Pixels that are considered in the disk.
        """
        if projection.lower() == 'sky':
            mask = np.logical_and(self.r_sky * self.dist >= self.r_min,
                                  self.r_sky * self.dist <= self.r_max)
        elif projection.lower() == 'disk':
            mask = np.logical_and(self.r_disk >= self.r_min,
                                  self.r_disk <= self.r_max)
        else:
            raise ValueError("Unknown projection {}.".format(projection)
                             + " Must be 'disk' or 'sky'.")
        return mask

    def set_linewidth(self, dV0=None, dVq=None, dVmax=None):
        """
        Set the radial linewidth profile in [m/s].
        """
        if dV0 is None:
            dV0 = (2. * sc.k * self.Tb0 / self.mu / sc.m_p)**0.5
        if dVq is None:
            dVq = 0.5 * self.Tbq
        self.dV0, self.dVq = dV0, dVq
        self.dV = simple_disk.powerlaw(self.r_sky * self.dist / 100.,
                                       self.dV0, self.dVq)
        if dVmax is not None:
            self.dV = np.where(self.dV <= dVmax, self.dV, dVmax)

    def set_brightness(self, Tb0, Tbq, Tbmax=None, r_min=None, r_max=None):
        """
        Set the radial brightness temperature profile in [K].
        """
        self.Tb0, self.Tbq = Tb0, Tbq
        self.r_min = self.r_min if r_min is None else r_min
        self.r_max = self.r_max if r_max is None else r_max
        self.Tb = simple_disk.powerlaw(self.r_sky * self.dist / 100., Tb0, Tbq)
        if Tbmax is not None:
            self.Tb = np.where(self.Tb <= Tbmax, self.Tb, Tbmax)
        self.Tb = np.where(self.in_disk(projection='sky'), self.Tb, 0.0)

    def interpolate_model(self, radii, model, parameter, radii_unit='au',
                          interp1d_kwargs=None):
        """
        Interpolate a user-provided model for the brightness temperature
        profile or the line width.

        Args:
            radii (array): Array of radii at which the model is sampled at in
                units given by ``radii_units``, either ``'au'`` or
                ``'arcsec'``.
            model (array): Array of model values evaluated at ``radii``. If
                brightness temperature, in units of [K], or if for linewidth,
                units of [m/s].
            parameter (str): Parameter of the model, either ``'Tb'`` for
                brightness temperature, or ``'dV'`` for linewidth.
            radii_unit (Optional[str]): Unit of the radii array, either
                ``'au'`` or ``'arcsec'``.
            interp1d_kwargs (Optional[dict]): Dictionary of kwargs to pass to
                ``intep1d`` used for the linear interpolation.
        """
        from scipy.interpolate import interp1d
        if radii.size != model.size:
            raise ValueError("`radii.size` does not equal `model.size`.")
        if radii_unit.lower() == 'au':
            radii /= self.dist
        elif radii_unit.lower() != 'arcsec':
            raise ValueError("Unknown `radii_unit` {}.".format(radii_unit))
        ik = {} if interp1d_kwargs is None else interp1d_kwargs
        ik['bounds_error'] = ik.pop('bounds_error', False)
        ik['fill_value'] = ik.pop('fill_value', 'extrapolate')
        ik['assume_sorted'] = ik.pop('assume_sorted', False)
        if parameter.lower() == 'tb':
            self.Tb = interp1d(radii, model, **ik)(self.r_sky)
            self.Tb = np.where(self.Tb < 0.0, 0.0, self.Tb)
            self.Tb0, self.Tbq = np.nan, np.nan
        elif parameter.lower() == 'dv':
            self.dV = interp1d(radii, model, **ik)(self.r_sky)
            self.dV = np.where(self.dV < 0.0, 0.0, self.dV)
            self.dV0, self.dVq = np.nan, np.nan

    def _calculate_vkep(self, rvals, tvals, zvals=0.0, inc=90.0):
        """
        Calculate the Keplerian velocity field, including vertical shear.

        Args:
            rvals (array): Midplane radii in [au].
            tvals (array): Midplane polar angles in [radians].
            zvals (Optional[array]): Height of emission surface in [au].
            inc (Optional[float]): Inclination of the disk in [degrees].

        Returns:
            vkep (array): Projected velocity in [m/s].
        """
        zvals = zvals if zvals is not None else np.zeros(rvals.shape)
        vkep2 = sc.G * self.mstar * self.msun * rvals**2.0
        vkep2 /= np.hypot(rvals, zvals)**3.0
        vkep = np.sqrt(vkep2 / sc.au)
        return vkep * np.cos(tvals) * np.sin(np.radians(inc))

    @property
    def vkep_sky(self):
        """
        Projected Keplerian rotation.
        """
        return self._calculate_vkep(self.r_sky * self.dist, self.t_sky,
                                    self.z_sky * self.dist, self.inc)

    @property
    def vkep_disk(self):
        """
        Disk-frame Keplerian rotation.
        """
        return self._calculate_vkep(self.r_disk, 0.0 * self.t_disk)

    # -- Deprojection Functions -- #

    def disk_coords(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, z0=0.0, psi=0.0,
                    z1=0.0, phi=0.0, frame='cylindrical'):
        r"""
        Get the disk coordinates given certain geometrical parameters and an
        emission surface. The emission surface is parameterized as a powerlaw
        profile:

        .. math::

            z(r) = z_0 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\psi} +
            z_1 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\varphi}

        Where both ``z0`` and ``z1`` are given in [arcsec]. For a razor thin
        disk, ``z0=0.0``, while for a conical disk, ``psi=1.0``. Typically
        ``z1`` is not needed unless the data is exceptionally high SNR and well
        spatially resolved.

        It is also possible to override this parameterization and directly
        provide a user-defined ``z_func``. This allow for highly complex
        surfaces to be included. If this is provided, the other height
        parameters are ignored.

        Args:
            x0 (Optional[float]): Source right ascension offset [arcsec].
            y0 (Optional[float]): Source declination offset [arcsec].
            inc (Optional[float]): Source inclination [deg].
            PA (Optional[float]): Source position angle [deg]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (Optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (Optional[float]): Flaring angle for the emission surface.
            z1 (Optional[float]): Aspect ratio correction term at 1" for the
                emission surface. Should be opposite sign to ``z0``.
            phi (Optional[float]): Flaring angle correction term for the
                emission surface.
            frame (Optional[str]): Frame of reference for the returned
                coordinates. Either ``'polar'`` or ``'cartesian'``.

        Returns:
            Three coordinate arrays, either the cylindrical coordaintes,
            ``(r, theta, z)`` or cartestian coordinates, ``(x, y, z)``,
            depending on ``frame``.
        """

        # Check the input variables.

        frame = frame.lower()
        if frame not in ['cylindrical', 'cartesian']:
            raise ValueError("frame must be 'cylindrical' or 'cartesian'.")

        # Define the emission surface function. Use the simple double
        # power-law profile.

        def z_func(r):
            z = z0 * np.power(r, psi) + z1 * np.power(r, phi)
            if z0 >= 0.0:
                return np.clip(z, a_min=0.0, a_max=None)
            return np.clip(z, a_min=None, a_max=0.0)

        # Calculate the pixel values.
        r, t, z = self._get_flared_coords(x0, y0, inc, PA, z_func)
        if frame == 'cylindrical':
            return r, t, z
        return r * np.cos(t), r * np.sin(t), z

    @staticmethod
    def _rotate_coords(x, y, PA):
        """Rotate (x, y) by PA [deg]."""
        x_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
        y_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
        return x_rot, y_rot

    @staticmethod
    def _deproject_coords(x, y, inc):
        """Deproject (x, y) by inc [deg]."""
        return x, y / np.cos(np.radians(inc))

    def _get_cart_sky_coords(self, x0, y0):
        """Return caresian sky coordinates in [arcsec, arcsec]."""
        return self.x_sky - x0, self.y_sky - y0

    def _get_polar_sky_coords(self, x0, y0):
        """Return polar sky coordinates in [arcsec, radians]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        return np.hypot(y_sky, x_sky), np.arctan2(x_sky, y_sky)

    def _get_midplane_cart_coords(self, x0, y0, inc, PA):
        """Return cartesian coordaintes of midplane in [arcsec, arcsec]."""
        x_sky, y_sky = self._get_cart_sky_coords(x0, y0)
        x_rot, y_rot = simple_disk._rotate_coords(x_sky, y_sky, PA)
        return simple_disk._deproject_coords(x_rot, y_rot, inc)

    def _get_midplane_polar_coords(self, x0, y0, inc, PA):
        """Return the polar coordinates of midplane in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)

    def _get_flared_coords(self, x0, y0, inc, PA, z_func):
        """Return cylindrical coordinates of surface in [arcsec, radians]."""
        x_mid, y_mid = self._get_midplane_cart_coords(x0, y0, inc, PA)
        r_tmp, t_tmp = np.hypot(x_mid, y_mid), np.arctan2(y_mid, x_mid)
        for _ in range(5):
            y_tmp = y_mid + z_func(r_tmp) * np.tan(np.radians(inc))
            r_tmp = np.hypot(y_tmp, x_mid)
            t_tmp = np.arctan2(y_tmp, x_mid)
        return r_tmp, t_tmp, z_func(r_tmp)

    @property
    def xaxis_disk(self):
        return self.x_disk[0]

    @property
    def yaxis_disk(self):
        return self.y_disk[:, 0]

    @property
    def xaxis_sky(self):
        return self.x_sky[0]

    @property
    def yaxis_sky(self):
        return self.y_sky[:, 0]

    # -- Pseudo Images -- #

    def get_channel(self, v_min, v_max, dv0=0.0, bmaj=None, bmin=None, bpa=0.0,
                    rms=0.0):
        """
        Calculate the channel emission in [K]. Can include velocity
        perturbations with the `dv0` parameter. To simulate observations this
        can include convolution with a 2D Gaussian beam or the addition of
        (correlated) noise.

        Args:
            v_min (float): The minimum velocity of the channel in [m/s].
            v_max (float): The maximum velocity of the channel in [m/s].
            dv0 (optional[ndarray]): An array of projected velocity
                perturbations in [m/s].
            bmaj (optional[float]): Synthesised beam major axis in [arcsec].
            bmin (optional[float]): Synthesised beam minor axis in [arcsec]. If
                only `bmaj` is specified, will assume a circular beam.
            bpa (optional[float]): Beam position angle in [deg].
            rms (optional[float]): RMS of the noise to add to the image.

        Returns:
            channel (ndarray): A synthesied channel map in [K].
        """

        # Check the channel boundaries are OK.

        v_max = np.median(self.dV) if v_max is None else v_max
        v_min = -v_max if v_min is None else v_min

        # Calculate the flux.

        v0 = self.vkep_sky + dv0
        flux = self.Tb * np.pi**0.5 * self.dV / 2.0 / (v_max - v_min)
        flux *= erf((v0 - v_min) / self.dV) - erf((v0 - v_max) / self.dV)

        # Include a beam convolution if necessary.

        beam = None if bmaj is None else self._get_beam(bmaj, bmin, bpa)
        if beam is not None:
            flux = convolve(flux, beam)

        # Add noise and return.

        noise = np.random.randn(flux.size).reshape(flux.shape)
        if beam is not None:
            noise = convolve(noise, beam)
        noise *= rms / np.std(noise)
        return flux + noise

    def get_cube(self, velax, dv0=0.0, bmaj=None, bmin=None, bpa=0.0, rms=0.0,
                 spectral_response=None):
        """
        Return the pseudo-cube with the given velocity axis.

        Args:
            velax (array): 1D array of channel centres in [m/s].
            dv0 (optional[ndarray]): An array of projected velocity
                perturbations in [m/s].
            bmaj (optional[float]): Synthesised beam major axis in [arcsec].
            bmin (optional[float]): Synthesised beam minor axis in [arcsec]. If
                only `bmaj` is specified, will assume a circular beam.
            bpa (optional[float]): Beam position angle in [deg].
            rms (optional[float]): RMS of the noise to add to the image.
            spectral_response (optional[list]): The kernel to convolve the cube
                with along the spectral dimension to simulation the spectral
                response of the telescope.

        Returns:
            cube (array): A 3D image cube.
        """
        # Define the velocity axis.

        vchan = abs(np.diff(velax)).mean()
        vbins = np.linspace(velax[0] - 0.5 * vchan,
                            velax[-1] + 0.5 * vchan,
                            velax.size + 1)

        # Make the image cube.

        cube = np.array([self.get_channel(vbins[i], vbins[i+1], dv0)
                         for i in range(velax.size)])
        assert cube.shape[0] == velax.size, "not all channels created"

        # Include convolution.

        beam = self._get_beam(bmaj, bmin, bpa) if bmaj is not None else None
        if beam is not None:
            cube = simple_disk._convolve_cube(cube, beam)
        if spectral_response is not None:
            cube = np.convolve(cube, spectral_response, axis=0)

        # Add noise and return.

        if rms > 0.0:
            noise = np.random.randn(cube.size).reshape(cube.shape)
            if beam is not None:
                noise = simple_disk._convolve_cube(noise, beam)
            if spectral_response is not None:
                noise = np.convolve(noise, spectral_response, axis=0)
            noise *= rms / np.std(noise)
        else:
            noise = np.zeros(cube.shape)
        return cube + noise

    @staticmethod
    def _convolve_cube(cube, beam):
        """Convolve the cube."""
        return np.array([convolve(c, beam) for c in cube])

    def _get_beam(self, bmaj, bmin=None, bpa=0.0):
        """Make a 2D Gaussian kernel for convolution."""
        bmin = bmaj if bmin is None else bmin
        bmaj /= self.cell_sky * self.fwhm
        bmin /= self.cell_sky * self.fwhm
        return Gaussian2DKernel(bmin, bmaj, np.radians(bpa))

    # -- Velocity Perturbations -- #

    def _perturbation(self, r0, t0, dr, dt=0.0, beta=0.0, projection='sky',
                      trim_values=False):
        """
        Define a velocity perturbation in cylindrical coordinates in either
        sky-plane coordaintes, ``projection='sky'``, or disk plane coordinates,
        ``projection='disk'``. If ``dt`` is set to zero, it assumes an
        azimuthally symmetric perturbation.

        Args:
            r0 (float): Radius of perturbation center. If ``projection='sky'``
                this is in [arcsec], while for ``projection='disk'`` this is in
                [au].
            t0 (float): Polar angle in [degrees] of perturbation center.
            dr (float): Radial width of perturbation. If ``projection='sky'``
                this is in [arcsec], while for ``projection='disk'`` this is in
                [au].
            dt (Optional[float]): Azimuthal extent of perturbations in [deg].
            beat (Optional[float]): Fixed pitch angle in [deg].
            projection (Optional[str]): If ``'sky'``, return the function in
                sky coordinates, otherwise in disk coordinates.
            trim_values(Optional[float]): If a number is specfied, fill all
                absolute values below this as ``np.nan``, primarily used for
                plotting.

        Returns:
            f (array): 2D array of the Gaussian perturbation.
        """

        # Parse input variables.

        if projection.lower() == 'sky':
            rvals, tvals = self.r_sky, self.t_sky
        elif projection.lower() == 'disk':
            rvals, tvals = self.r_disk, self.t_disk
        else:
            raise ValueError("`projection` must be 'sky' or 'disk'.")
        if dt == 0.0 and beta != 0.0:
            raise ValueError("Cannot specify pitch angle and `dt=0.0`.")

        # Azimuthally symmetric perturbation.

        if dt == 0.0:
            return np.exp(-0.5*((rvals - r0) / dr)**2.0)

        # Calculate azmithal dependance.

        f = []
        nwrap = self.nwrap if self.nwrap % 2 else self.nwrap + 1
        for wrap in np.arange(nwrap) - (nwrap - 1) / 2:

            t_tmp = tvals.copy() + wrap * 2.0 * np.pi
            r0_tmp = r0 / (1.0 + t_tmp * np.tan(np.radians(beta)))
            t_tmp -= np.radians(t0)

            _f = np.exp(-0.5*((rvals - r0_tmp) / dr)**2.0)
            f += [_f * np.exp(-0.5*(t_tmp / np.radians(dt))**2.0)]
        f = np.sum(f, axis=0)

        # Apply trims.

        if trim_values:
            f = np.where(abs(f) > trim_values, f, np.nan)

        # Return

        return f

    def radial_perturbation(self, dv, r0, t0, dr, dt=0.0, beta=0.0,
                            projection='sky', trim_values=False):
        """
        Gaussian perturbation with radial velocity projection.
        """
        f = dv * self._perturbation(r0=r0, t0=t0, dr=dr, dt=dt, beta=beta,
                                    projection=projection,
                                    trim_values=trim_values)
        if projection.lower() == 'disk':
            return f
        return f * np.sin(self.t_sky) * np.sin(self.inc)

    def rotational_perturbation(self, dv, r0, t0, dr, dt=0.0, beta=0.0,
                                projection='sky', trim_values=False):
        """
        Gaussian perturbation with rotational velocity projection.
        """
        f = dv * self._perturbation(r0=r0, t0=t0, dr=dr, dt=dt, beta=beta,
                                    projection=projection,
                                    trim_values=trim_values)
        if projection.lower() == 'disk':
            return f
        return f * np.cos(self.t_sky) * np.sin(self.inc)

    def vertical_perturbation(self, dv, r0, t0, dr, dt=0.0, beta=0.0,
                              projection='sky', trim_values=False):
        """
        Gaussian perturbation with vertical velocity projection.
        """
        f = dv * self._perturbation(r0=r0, t0=t0, dr=dr, dt=dt, beta=beta,
                                    projection=projection,
                                    trim_values=trim_values)
        if projection.lower() == 'disk':
            return f
        return f * np.cos(self.inc)

    def doppler_flip(self, dv, r0, t0, dr, dt, beta=0.0, dr0=0.5, dt0=1.0,
                     clockwise=True, projection='sky', trim_values=False):
        """
        Simple 'Doppler flip' model with two offset azimuthal deviations.

        Args:
            v (float): Azimuthal velocity deviation in [m/s].
            r0 (float): Radius in [au] of Doppler flip center.
            t0 (float): Polar angle in [degrees] of Doppler flip center.
            dr (float): Radial width of each Gaussian in [au].
            dt (float): Azimuthal width (arc length) of each Gaussian in [au].
            dr0 (Optional[float]): Relative radial offset between the positive
                and negative lobes. Defaults to 0.5.
            dt0 (Optional[float]): Relative azimuthal offset between the
                positive and negative lobes. Defaults to 1.0.

        Returns:
            dv0 (array): Array of velocity devitiations in [m/s]. If
                ``projection='sky'``, these will be projected on the sky.
        """
        rp = r0 + dr0 * dr
        rn = r0 - dr0 * dr
        dt0 /= self.dist if projection.lower() == 'sky' else 1.0
        tp = t0 + np.degrees(dt0 * dt / rp)
        tn = t0 - np.degrees(dt0 * dt / rn)
        if not clockwise:
            temp = tn
            tn = tp
            tp = temp
            beta = -beta
        vp = self.rotational_perturbation(dv=dv, r0=rp, t0=tp, dr=dr, dt=dt,
                                          beta=-beta, projection=projection)
        vn = self.rotational_perturbation(dv=dv, r0=rn, t0=tn, dr=dr, dt=dt,
                                          beta=-beta, projection=projection)
        v = vp - vn
        if trim_values:
            v = np.where(abs(v) > trim_values, v, np.nan)
        return v

    def radial_doppler_flip(self, dv, r0, t0, dr, dt, dr0=0.5, dt0=1.0,
                            flip_rotation=False, projection='sky',
                            trim_values=False):
        """
        Simple `Doppler flip` model but with radial velocity deviations intead.

        Args:
            dv (float): Radial velocity deviation in [m/s].
            r0 (float): Radius in [au] of Doppler flip center.
            t0 (float): Polar angle in [degrees] of Doppler flip center.
            dr (float): Radial width of each Gaussian in [au].
            dt (float): Azimuthal width (arc length) of each Gaussian in [au].
            dr0 (Optional[float]): Relative radial offset between the positive
                and negative lobes. Defaults to 0.5.
            dt0 (Optional[float]): Relative azimuthal offset between the
                positive and negative lobes. Defaults to 1.0.

        Returns:
            dv0 (array): Array of velocity devitiations in [m/s]. If
                ``sky=True``, these will be projected on the sky.
        """
        rp = r0 + dr0 * dr
        rn = r0 - dr0 * dr
        tp = t0 + np.degrees(dt0 * dt / rp)
        tn = t0 - np.degrees(dt0 * dt / rn)
        if flip_rotation:
            temp = tn
            tn = tp
            tp = temp
        vp = self.radial_perturbation(dv=dv, r0=rp, t0=tp, dr=dr, dt=dt,
                                      projection=projection)
        vn = self.radial_perturbation(dv=dv, r0=rn, t0=tn, dr=dr, dt=dt,
                                      projection=projection)
        v = vp - vn
        if trim_values:
            v = np.where(abs(v) > trim_values, v, np.nan)
        return v

    def vertical_flow(self, v, r0, t0, dr, dt):
        return

    # -- Plotting Routines -- #

    def plot_keplerian(self, fig=None, logy=True, top_axis=True):
        """
        Plot the Keplerian rotation profile.
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        x = self.r_disk.flatten()
        y = self.vkep_disk.flatten()
        idxs = np.argsort(x)
        ax.plot(x[idxs], y[idxs])
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('Keplerian Rotation [m/s]')
        if logy:
            ax.set_yscale('log')
        if top_axis:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] / self.dist,
                         ax.set_xlim()[1] / self.dist)
            ax2.set_xlabel('Radius [arcsec]')

    def plot_linewidth(self, fig=None, top_axis=True):
        """
        Plot the linewidth profile.
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        x = self.r_sky.flatten() * self.dist
        y = self.dV.flatten()
        idxs = np.argsort(x)
        ax.plot(x[idxs], y[idxs])
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('Doppler Linewidth [m/s]')
        if top_axis:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] / self.dist,
                         ax.set_xlim()[1] / self.dist)
            ax2.set_xlabel('Radius [arcsec]')

    def plot_brightness(self, fig=None, top_axis=True):
        """
        Plot the brightness temperature profile.
        """
        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        x = self.r_sky.flatten() * self.dist
        y = self.Tb.flatten()
        idxs = np.argsort(x)
        ax.plot(x[idxs], y[idxs])
        ax.set_xlabel('Radius [au]')
        ax.set_ylabel('BrightestTemperature [K]')
        if top_axis:
            ax2 = ax.twiny()
            ax2.set_xlim(ax.get_xlim()[0] / self.dist,
                         ax.set_xlim()[1] / self.dist)
            ax2.set_xlabel('Radius [arcsec]')

    def plot_radii(self, ax, rvals, contour_kwargs=None, projection='sky'):
        """
        Plot annular contours onto the axis.
        """
        contour_kwargs = {} if contour_kwargs is None else contour_kwargs
        contour_kwargs['colors'] = contour_kwargs.pop('colors', '0.6')
        contour_kwargs['linewidths'] = contour_kwargs.pop('linewidths', 0.5)
        contour_kwargs['linestyles'] = contour_kwargs.pop('linestyles', '--')
        if projection.lower() == 'sky':
            x, y, z = self.x_sky[0], self.y_sky[:, 0], self.r_sky
        elif projection.lower() == 'disk':
            x, y, z = self.x_disk, self.y_disk, self.r_disk
        ax.contour(x, y, z, rvals, **contour_kwargs)

    @staticmethod
    def format_sky_plot(ax):
        """
        Default formatting for sky image.
        """
        ax.set_xlim(ax.get_xlim()[1], ax.get_xlim()[0])
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
        ax.set_xlabel('Offset [arcsec]')
        ax.set_ylabel('Offset [arcsec]')
        ax.scatter(0, 0, marker='x', color='0.7', lw=1.0, s=4)

    @staticmethod
    def format_disk_plot(ax):
        """
        Default formatting for disk image.
        """
        ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
        ax.set_xlabel('Offset [au]')
        ax.set_ylabel('Offset [au]')
        ax.scatter(0, 0, marker='x', color='0.7', lw=1.0, s=4)

    @staticmethod
    def BuRd():
        """Blue-Red color map."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        c2 = plt.cm.Reds(np.linspace(0, 1, 32))
        c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
        colors = np.vstack((c1, np.ones(4), c2))
        return mcolors.LinearSegmentedColormap.from_list('BuRd', colors)

    @staticmethod
    def RdBu():
        """Red-Blue color map."""
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        c2 = plt.cm.Reds(np.linspace(0, 1, 32))
        c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
        colors = np.vstack((c1, np.ones(4), c2))[::-1]
        return mcolors.LinearSegmentedColormap.from_list('RdBu', colors)

    @property
    def extent_sky(self):
        return [self.x_sky[0, 0],
                self.x_sky[0, -1],
                self.y_sky[0, 0],
                self.y_sky[-1, 0]]

    @property
    def extent_disk(self):
        return [self.x_sky[0, 0] * self.dist,
                self.x_sky[0, -1] * self.dist,
                self.y_sky[0, 0] * self.dist,
                self.y_sky[-1, 0] * self.dist]

    @staticmethod
    def powerlaw(x, x0, q):
        return x0 * np.power(x, q)
