#! /usr/bin/env python

"""This module implements the container classes for KIDs data."""

from astropy.nddata import NDDataRef
import astropy.units as u
from .utils import ExtendedNDDataRef, FrequencyDivisionMultiplexingDataRef
import numpy as np
from tollan.utils.log import timeit, get_logger


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
    extra_attrs : dict, optional
        A dict of extra data to attach to this sweep. They has to be
        of the same shape as `frequency`.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    _extra_attrs_to_slice = None

    def _slice_extra(self, item):
        # this returns extra sliced attributes when slicing
        result = {
                '_frequency': self._frequency[item],
                '_D21': self._D21[item],
                }
        if self._extra_attrs_to_slice is not None:
            for a in self._extra_attrs_to_slice:
                result[a] = getattr(self, a)[item]
        return result

    def __init__(
            self,
            S21=None, frequency=None, D21=None,
            extra_attrs=None,
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
        elif D21 is not None:
            kwargs['data'] = D21.data
            kwargs['unit'] = D21.unit

        super().__init__(**kwargs)

        # handle extra attrs
        if extra_attrs is not None:
            # add extra data objects
            if any(hasattr(self, a) for a in extra_attrs.keys()):
                raise ValueError(
                        "name of extra_attrs conflicts with existing attts.")
            extra_attrs_to_slice = list()
            for k, v in extra_attrs.items():
                if v.shape != self.data.shape:
                    raise ValueError("invalid shape of extra attr")
                setattr(self, k, v)
                extra_attrs_to_slice.append(k)
            self._extra_attrs_to_slice = extra_attrs_to_slice


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
        else:
            frequency = SweepMixin._validate_frequency(frequency)

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

    @property
    def unified(self):
        """The associated unified `Sweep` object."""
        return self._unified

    def set_unified(self, sweep):
        """Set the associated unified `Sweep` object.

        Parameters
        ----------
        sweep : `Sweep`
            The `Sweep` object that contains the channel merged data.
        """
        self._unified = sweep

    def make_unified(self, cached=True, *args, **kwargs):
        """Create unified sweep with D21 spectrum.

        Parameters
        ----------
        cached : bool, optional
            If True and D21 exists, it is returned.
        args, kwargs :
            The argument passed to `_make_D21`.
        """
        if not (cached and hasattr(self, '_unified')):
            self.set_unified(self._make_unified(*args, **kwargs))
        return self.unified

    @timeit
    def _make_unified(
            self, flim=None, fstep=None, resample=None,
            exclude_edge_samples=10,
            smooth=11, method='savgol'):
        """Compute the unified ``D21`` spectrum.

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
            Apply smooth to the IQs for D21 computation.
        method: 'savgol' or 'gradient'
            The method for D21 compuatation.
        """
        logger = get_logger()
        if fstep is None and resample is None:
            fstep = 1000. << u.Hz
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
                f" original fs=[{fs.min()}, {fs.max()}] smooth={smooth} method={method}")
        fs = np.arange(
                fmin.to_value(u.Hz),
                fmax.to_value(u.Hz),
                fstep.to_value(u.Hz)) << u.Hz
        adiqs0 = np.abs(self.diqs_df(self.S21, self.frequency, smooth=smooth, method=method))
        adiqs = np.zeros(fs.shape, dtype=np.double) << u.adu / u.Hz
        adiqscov = np.zeros(fs.shape, dtype=int)
        if exclude_edge_samples > 0:
            es = slice(exclude_edge_samples, -exclude_edge_samples)
        else:
            es = slice(None)

        for i in range(self.frequency.shape[0]):
            m = (fs >= self.frequency[i].min()
                    ) & (fs <= self.frequency[i].max())
            tmp = np.interp(
                    fs[m], self.frequency[i, es], adiqs0[i, es],
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
        unified = Sweep(
                S21=None, D21=adiqs, frequency=fs,
                extra_attrs={
                    'd21_cov': adiqscov,
                    }
                )
        return unified

    @staticmethod
    def diqs_df(iqs, fs, smooth=None, method='gradient'):
        if smooth in (None, 0):
            if method != 'gradient':
                raise ValueError("no-smooth only works for gradient")
            else:
                pass
        diqs = np.empty(iqs.shape, dtype=iqs.dtype) << (u.adu / u.Hz)
        if method == 'gradient':
            if smooth is not None and smooth > 0:
                def csmooth(arr, *args, **kwargs):
                    from scipy.ndimage.filters import uniform_filter1d
                    arr_r = uniform_filter1d(arr.real, *args, **kwargs)
                    arr_i = uniform_filter1d(arr.imag, *args, **kwargs)
                    return arr_r + 1.j * arr_i
                iqs = csmooth(iqs, size=smooth, mode='mirror') << u.adu
            for i in range(iqs.shape[0]):
                diqs[i] = np.gradient(iqs[i], fs[i])
        elif method == 'savgol':
            from scipy.signal import savgol_filter
            for i in range(iqs.shape[0]):
                df = (fs[i][1] - fs[i][0]).to_value(u.Hz)
                xx = savgol_filter(
                        iqs[i].real.to_value(u.adu),
                        window_length=smooth,
                        polyorder=2, deriv=1, delta=df)
                yy = savgol_filter(
                        iqs[i].imag.to_value(u.adu),
                        window_length=smooth,
                        polyorder=2, deriv=1, delta=df)
                diqs[i] = (xx + 1.j * yy) << u.adu / u.Hz
        return diqs

    def get_sweep(self, tone_id, **kwargs):
        """Return a `Sweep` object for a single channel."""

        s = slice(tone_id, tone_id + 1)
        return Sweep(
                frequency=self.frequency[tone_id],
                S21=self.S21[tone_id],
                D21=np.abs(self.diqs_df(
                    self.S21[s], self.frequency[s], **kwargs)[0]))

    def __str__(self):
        if self.data is None:
            shape = '(empty)'
        else:
            shape = self.data.shape
        return f'{self.__class__.__name__}{shape}'
