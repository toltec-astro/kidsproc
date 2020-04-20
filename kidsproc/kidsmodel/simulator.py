#! /usr/bin/env python
import numpy as np
# from astropy import units as u
from astropy.modeling import models
from astropy.modeling import fix_inputs
from . import (
        # ResonanceCircleComplex,
        ResonanceCircleInv,
        ResonanceCircleComplex,
        ResonanceCircleComplexInv,
        ResonanceCircleSweepComplex,
        ResonanceCircleProbeComplex,
        ReadoutIQToComplex,
        OpticalDetune,
        InstrumentalDetune,
        ResonanceCircleQrInv
        )
from tollan.utils.log import get_logger


class KidsSimulator(object):
    """Class that make simulated kids data."""

    logger = get_logger()

    def __init__(self, fr=None, Qr=None,
                 background=None, responsivity=None):
        self._fr = fr
        self._Qr = Qr
        self._background = background
        self._responsivity = responsivity
        self._n_models = self._fr.shape[0]

        m_info = ['summary of kids simulator models:', ]
        sep = '-*' * 40
        m_info.append(f"{sep}\nx sweep:\n{self._x_sweep}")
        m_info.append(f"{sep}\nf sweep:\n{self._f_sweep}")
        m_info.append(f"{sep}\nx probe:\n{self._x_probe}")
        m_info.append(f"{sep}\nf_probe:\n{self._f_probe}")
        m_info.append(f"{sep}\np_probe:\n{self._p_probe}")
        m_info.append(f"{sep}")
        self.logger.info('\n'.join(m_info))

    @property
    def _Qr2r(self):
        """Model to convert Qr to r."""
        return ResonanceCircleQrInv(n_models=self._n_models)

    @property
    def _combine_readout(self):
        """Model to combine the readout to complex values."""
        return ReadoutIQToComplex(n_models=self._n_models)

    @property
    def _rx2iqcomplex(self):
        """Model to convert rx to iq in complex."""
        return ResonanceCircleComplex(n_models=self._n_models)

    @property
    def _iq2rxcomplex(self):
        """Model to convert iq to rx in complex."""
        return ResonanceCircleComplexInv(n_models=self._n_models)

    @property
    def _iq2rx(self):
        """Model to convert iq to rx in separate channels."""
        return ResonanceCircleInv(n_models=self._n_models)

    @property
    def _fp2x(self):
        """Model to evaluate x at fp for given fr."""
        return fix_inputs(
                InstrumentalDetune(n_models=self._n_models),
                {'fr': self._fr})

    @property
    def _p2x(self):
        """Model to evaluate x at p for given background and
        responsivity."""
        return OpticalDetune(
                background=self._background,
                responsivity=self._responsivity,
                n_models=self._n_models)

    @property
    def _x2iq(self):
        """Model to evaluate IQ at x for given Qr."""
        return fix_inputs(
                (self._Qr2r & models.Scale(
                    np.ones((self._n_models, 1), dtype=float),
                    n_models=self._n_models)) |
                self._combine_readout | self._rx2iqcomplex,
                {
                    'Qr': self._Qr
                    })

    @property
    def _x_sweep(self):
        """Alias of `_x2iq`"""
        return self._x2iq

    @property
    def _f_sweep(self):
        """Model to evaluate IQ at f for given fr and Qr."""
        return ResonanceCircleSweepComplex(fr=self._fr, Qr=self._Qr)

    @property
    def _x_probe(self):
        """Alias of `_x2iq`"""
        return self._x2iq

    @property
    def _f_probe(self):
        """Model to evaluate IQ at (fr, Qr) for given fp."""
        return ResonanceCircleProbeComplex()

    @property
    def _p_probe(self):
        """Model to evaluate IQ at (Qr, p) for given background and
        responsivity."""
        return (self._Qr2r & self._p2x) | \
            self._combine_readout | self._rx2iqcomplex

    @property
    def fwhm_x(self):
        """Return the resonance FWHM in unit of x."""
        return 1. / self._Qr

    @property
    def fwhm_f(self):
        """Return the resonance FWHM in unit of Hz."""
        return self.fwhm_x * self._fr

    def sweep_x(self, n_steps=None, n_fwhms=None, xlim=None):
        """Return a resonance circle sweep."""
        if n_fwhms is None and xlim is None:
            raise ValueError("n_fwhms or xlim is required.")
        if xlim is None:
            xlim = (-0.5 * n_fwhms * self.fwhm_x,
                    0.5 * n_fwhms * self.fwhm_x)
        # get grid
        xs = np.linspace(*xlim, n_steps)
        iqs = self._x_sweep(xs)
        return xs, iqs

    def probe_p(self, pwrs, fp=None):
        """Return detector response for given optical power and probe
        frequency."""
        p2x = self._p2x
        p2x.n_models = self._n_models
        self.logger.debug(f"probe with n_mdoels={len(p2x)}")
        rs = np.full(pwrs.shape, self._Qr2r(self._Qr))
        xs = p2x(pwrs)
        if fp is not None:
            xs = xs + self._fp2x(fp)
        iqs = self._x_probe(xs)
        return rs, xs, iqs

    def solve_x(self, *args):
        """Return x for given detector response.
        """
        if len(args) == 1:
            # complex
            return self._iq2rxcomplex(args[0])
        return self._iq2rx(*args)
