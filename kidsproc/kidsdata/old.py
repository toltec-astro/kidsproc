#! /usr/bin/env python

"""This module implements the container classes for KIDs data."""

import numpy as np
from astropy.nddata import NDDataRef
from .timestream_mixin import TimeStreamMixin
from .sweep_mixin import SweepMixin
from cached_property import cached_property
from tollan.utils.log import get_logger, timeit
import astropy.units as u


class Sweep(SweepMixin, NDDataRef):
    """Container for frequency sweep data."""

    def __repr__(self):
        return f"{self.__class__.__name__}{self.data.shape}"

    @cached_property
    def fs(self):
        """The frequency grid in Hz."""
        return np.tile(
            self.meta['sweeps']['flo'].T,
            (len(self.meta['tones']), 1)
            ) + self.meta['tones']['fc'][:, None]

    @cached_property
    def iqs(self):
        """The raw (I, Q) values from readout."""
        return self.data

    _d21 = None

    def d21(self, **kwargs):
        # if self._d21 is None or len(kwargs) > 0:
        if self._d21 is None:
            self._d21 = self._make_d21(**kwargs)
        fs, adiqs, adiqscov = self._d21
        if hasattr(fs, 'unit'):
            return fs, adiqs, adiqscov
        return fs * u.Hz, adiqs, adiqscov

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

        if flim is None:
            flim = (self.fs.min(), self.fs.max())
        fmin, fmax = flim
        if resample is not None:
            fstep = (self.fs[0, 1] - self.fs[0, 0]) / resample

        logger.debug(
                f"build d21 with fs=[{fmin}, {fmax}, {fstep}]"
                f" exclude_edge_samples={exclude_edge_samples}"
                f" original fs=[{self.fs.min()}, {self.fs.max()}]")
        fs = np.arange(fmin, fmax, fstep)
        adiqs0 = np.abs(self.diqs_df(self.iqs, self.fs, smooth=smooth))
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
        return fs, adiqs, adiqscov

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


class VnaSweep(Sweep):
    pass


class TargetSweep(Sweep):
    pass


class TimeStream(TimeStreamMixin, NDDataRef):
    """Container for time stream data."""
    pass


class RawTimeStream(TimeStream):
    """Container for the raw in-phase (i) and quadrature (q) data."""
    pass


class SolvedTimeStream(TimeStream):
    """Container for the solved detuning (x) and loss (r) data."""
    pass
