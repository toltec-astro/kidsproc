#! /usr/bin/env python

"""This module implements the container classes for KIDs data."""

from astropy.nddata import NDDataRef
import astropy.units as u
from .utils import ExtendedNDDataRef, FrequencyDivisionMultiplexingDataRef
import numpy as np
from tollan.utils.log import timeit


__all__ = ['Sweep', 'MultiSweep', ]


class SweepMixin(object):
    """A mixin class for frequency sweep data.

    """

    @property
    def S21(self):
        """The S21."""
        return self.data << self.unit

    @property
    def frequency(self):
        """The frequency."""
        return self._frequency.data << self._frequency.unit

    @property
    def D21(self):
        """The D21."""
        return self._D21.data << self._D21.unit

    @staticmethod
    def _validate_S21(S21):
        # note that the returned item here is a quantity
        # which is different from the other _validate_* methods.
        if S21.dtype != complex:
            raise ValueError('S21 has to be complex.')
        if isinstance(S21, u.Quantity):
            if S21.unit != u.adu:
                raise ValueError('S21 unit has to be adu.')
            return S21.to(u.adu)
        return S21 << u.adu

    @staticmethod
    def _validate_frequency(frequency):
        if not isinstance(frequency, u.Quantity):
            raise ValueError("frequency has to be a quantity with unit.")
        if not frequency.unit.is_equivalent(u.Hz):
            raise ValueError('invalid unit for frequency.')
        return NDDataRef(data=frequency.value, unit=frequency.unit)

    @staticmethod
    def _validate_D21(D21):
        if not isinstance(D21, u.Quantity):
            raise ValueError("D21 has to be a quantity with unit.")
        if not D21.unit.is_equivalent(u.adu / u.Hz):
            raise ValueError('invalid unit for D21.')
        return NDDataRef(data=D21.value, unit=D21.unit)


class Sweep(ExtendedNDDataRef, SweepMixin):
    """A container class for frequency sweep data.

    The response of the sweep is expressed as the ``S21`` parameter of the
    2-ports readout circuit, in its native analog-to-digit unit (ADU),
    as a function of the probing frequency.

    `S21` could also be a n-dim array for `Sweep`, in which case the
    data is for a collection of sweeps that share the same frequency grid.
    The data shall be organized such that the frequency varies along the last
    dimension. Note that this is different from `MultiSweep` in that
    for `MultiSweep`, each sweep has its own frequency grid.

    The sweep may also be provided with a `D21` spectrum, which is the
    magnitude of the derivative of the complex `S21` with respect to the
    frequency. The `D21` spectrum is a reduced form of `S21` thus does not
    carry all information that `S21` has, but it is easier to visualize
    when we only cares the resonance frequency and the quality factor.

    Parameters
    ----------
    S21 : `astropy.nddata.NDData`, `astropy.units.Quantity`
        The S21, in (or assumed to be in) ADU.
    frequency : `astropy.units.Quantity`
        The frequency.
    D21 : `astropy.units.Quantity`
        The D21 spectrum.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    def _slice_extra(self, item):
        # this returns extra sliced attributes when slicing
        return {
                '_frequency': self._frequency[item],
                '_D21': self._D21[item],
                }

    def __init__(
            self,
            S21=None, frequency=None, D21=None,
            **kwargs):

        if 'data' in kwargs:
            # In cases of slicing, new objects will be initialized with `data`
            # instead of ``S21``. Ensure we grab the `data` argument.
            if S21 is None:
                super().__init__(**kwargs)
                # additional attributes frequency and d21 will be added
                # by the _slice_extra call at the end of the slice
                # automatically
                return
            else:
                raise ValueError('data should not be specified.')

        # this is for normal construction
        # we expect S21 and frequency to be always set
        # and d21 is optional
        # check dimensions
        if S21 is not None:
            if frequency is None:
                raise ValueError("S21 requires frequency.")
            if frequency.ndim > 0 and frequency.shape[-1] != S21.shape[-1]:
                raise ValueError("shape of frequency does not match S21.")
            if D21 is not None:
                if frequency.ndim > 0 and frequency.shape[-1] != D21.shape[-1]:
                    raise ValueError("shape of frequency does not match D21.")

        if frequency is not None:
            frequency = self._validate_frequency(frequency)
            self._frequency = frequency

        if D21 is not None:
            D21 = self._validate_D21(D21)
            self._D21 = D21

        if S21 is not None:
            S21 = self._validate_S21(S21)
            kwargs['data'] = S21.data
            kwargs['unit'] = S21.unit

        super().__init__(**kwargs)


class _MultiSweepDataRef(FrequencyDivisionMultiplexingDataRef):
    """A generic container for FDM frequency sweep data.

    The FDM sweep data is generated by reading around multiple ``tones`` in the
    frequency space, with the sweep accomplished via the change of the local
    oscillator frequency at each sweep step (referred to as ``sweeps``).

    This class sets the generic structure of the data but the actual
    implementation is in the subclass `MultiSweep`.

    Parameters
    ----------
    tones : `astropy.units.Quantity`
        The tone frequencies.
    sweeps : `astropy.units.Quantity`
        The sweep frequencies.
    frequency : `astropy.units.Quantity`
        The frequency grid.
    data : `astropy.nddata.NDData`
        The data.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    def __init__(
            self, tones=None, sweeps=None, frequency=None,
            data=None, **kwargs):

        if frequency is None:
            if sum([tones is None, sweeps is None]) == 1:
                raise ValueError(
                    "must specify both tones and sweeps.")
            # create the frequency grid
            if tones is not None:
                frequency = self._make_frequency_grid(tones, sweeps)

        # check lengths of axis
        if data is not None and frequency is not None:
            if data.shape != frequency.data.shape:
                raise ValueError(
                    f"data of shape {data.shape} does not match the shape of"
                    f" the frequency grid {frequency.shape}.")

        super().__init__(tones=tones, data=data, **kwargs)
        self._sweeps = sweeps
        self._frequency = frequency

    @staticmethod
    def _make_frequency_grid(tones, sweeps):
        """Return the frequency array from the `tones` and `sweeps`."""
        data = sweeps[None, :] + tones[:, None]
        return NDDataRef(
                data=data.value,
                unit=data.unit, uncertainty=None,
                meta=None)

    @property
    def sweeps(self):
        """The sweeps."""
        return self._sweeps


class MultiSweep(_MultiSweepDataRef, SweepMixin):
    """A container class for multiplexed frequency sweep data.

    This class is different from the `Sweep` in that the S21 data is measured
    from a multiplexed readout system and is organized in a 2-d array where the
    first axis is for the tones, and the second axis is for the sweep steps.

    Parameters
    ----------
    S21 : `astropy.nddata.NDData`, `astropy.units.Quantity`
        The S21 data, in (or assumed to be in) ADU.
    tones : `astropy.units.Quantity`
        The tone frequencies.
    sweeps : `astropy.units.Quantity`
        The sweep frequencies.
    frequency : `astropy.units.Quantity`
        The frequency grid.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    def _slice_extra(self, item):
        # this returns extra sliced attributes when slicing
        result = {
                '_frequency': self._frequency[item],
                }
        tone_slice = None
        sweep_slice = None
        if not isinstance(item, tuple):
            # normalize to a tuple
            item = (item, )
        if len(item) == 1:
            tone_slice, = item
        elif len(item) == 2:
            tone_slice, sweep_slice = item
        else:
            raise ValueError("too many slices.")
        if tone_slice is not None:
            result['_tones'] = self._tones[tone_slice]
        else:
            result['_tones'] = self._tones
        if sweep_slice is not None:
            result['_sweeps'] = self._sweeps[sweep_slice]
        else:
            result['_sweeps'] = self._sweeps
        return result

    def __init__(
            self,
            S21=None, tones=None, sweeps=None, frequency=None,
            **kwargs):

        if 'data' in kwargs:
            # In cases of slicing, new objects will be initialized with `data`
            # instead of ``S21``. Ensure we grab the `data` argument.
            if S21 is None:
                super().__init__(**kwargs)
                # additional attributes frequency and d21 will be added
                # by the _slice_extra call at the end of the slice
                # automatically
                return
            else:
                raise ValueError('data should not be specified.')

        # this is for normal construction
        if S21 is not None:
            S21 = self._validate_S21(S21)
            kwargs['data'] = S21.value
            kwargs['unit'] = S21.unit

        super().__init__(
                tones=tones, sweeps=sweeps, frequency=frequency, **kwargs)

    @timeit
    def _make_d21(
            self, flim=None, fstep=None, resample=None,
            exclude_edge_samples=10,
            smooth=3):
        """The unified ``D21`` spectrum.

        Parameters
        ----------
        flim: tuple, optional
            If set, the d21 is resampled on to this frequency range.

        fstep: float, optional
            The step size of the frequency grid in Hz.

        resample: float, optional
            Alternative way to specify `fstep`. The frequency step
            used will be smaller than the input data by this factor.

        exclude_edge_samples: int
            Number of samples to exclude at the tone edges.

        smooth: int
            Apply smooth to the IQs *before* the gradient operation.
        """
        logger = get_logger()
        if fstep is None and resample is None:
            fstep = 1000.
        if not (bool(fstep is not None) ^ bool(resample is not None)):
            raise ValueError("only one of fstep or resample can be specified")

        fs = self.frequency
        if flim is None:
            flim = (fs.min(), fs.max())
        fmin, fmax = flim
        if resample is not None:
            fstep = (fs[0, 1] - fs[0, 0]) / resample

        logger.debug(
                f"build d21 with fs=[{fmin}, {fmax}, {fstep}]"
                f" exclude_edge_samples={exclude_edge_samples}"
                f" original fs=[{fs.min()}, {fs.max()}]")
        fs = np.arange(fmin, fmax, fstep)
        adiqs0 = np.abs(self.diqs_df(self.S21, self.frequency, smooth=smooth))
        adiqs = np.zeros(fs.shape, dtype=np.double)
        adiqscov = np.zeros(fs.shape, dtype=int)
        if exclude_edge_samples > 0:
            es = slice(exclude_edge_samples, -exclude_edge_samples)
        else:
            es = slice(None)

        for i in range(self.fs.shape[0]):
            m = (fs >= self.fs[i].min()) & (fs <= self.fs[i].max())
            tmp = np.interp(
                    fs[m], self.fs[i, es], adiqs0[i, es],
                    left=np.nan,
                    right=np.nan,
                    )
            cov = ~np.isnan(tmp)
            tmp[~cov] = 0
            tmp[cov] += adiqs[m][cov]
            adiqs[m] += tmp
            adiqscov[m] += cov.astype(dtype=int)
        m = adiqscov > 0
        adiqs[m] /= adiqscov[m]
        adiqs[np.isnan(adiqs)] = 0
        d21 = Sweep(
                S21=None, D21=adiqs, frequency=fs,
                )
        d21.coverage = adiqscov
        return d21

    @staticmethod
    def diqs_df(iqs, fs, smooth=None):
        if smooth in (None, 0):
            pass
        else:
            def csmooth(arr, *args, **kwargs):
                from scipy.ndimage.filters import uniform_filter1d
                arr_r = uniform_filter1d(arr.real, *args, **kwargs)
                arr_i = uniform_filter1d(arr.imag, *args, **kwargs)
                return arr_r + 1.j * arr_i
            iqs = csmooth(iqs, size=smooth, mode='mirror')
        diqs = np.empty_like(iqs)
        for i in range(iqs.shape[0]):
            diqs[i] = np.gradient(iqs[i], fs[i])
        return diqs
