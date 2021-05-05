#! /usr/bin/env python
import numpy as np
from astropy.modeling import Parameter, Model
from . import (
        # ResonanceCircleComplex,
        ResonanceCircleInv,
        ResonanceCircleComplex,
        ResonanceCircleComplexInv,
        # ResonanceCircleSweepComplex,
        # ResonanceCircleProbeComplex,
        ReadoutIQToComplex,
        OpticalDetune,
        InstrumentalDetune,
        ResonanceCircleQrInv,
        _Model
        )
from tollan.utils.log import get_logger


class Identity1D(_Model):

    n_inputs = 1
    n_outputs = 1

    @staticmethod
    def evaluate(x):
        return x

    def inverse(self):
        return self


class _x2rx(Model):

    n_inputs = 1
    n_outputs = 2

    Qr = Parameter(default=1e4)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.inputs = ('x', )
        self.outputs = ('r', 'x')
        self._Qr2r = ResonanceCircleQrInv()

    def evaluate(self, x, Qr):
        r = self._Qr2r(Qr)
        return np.tile(r, (x.shape[0], 1)), x.T


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

        # m_info = ['summary of kids simulator models:', ]
        # sep = '-*' * 40
        # m_info.append(f"{sep}\nx sweep:\n{self._x_sweep}")
        # m_info.append(f"{sep}\nf sweep:\n{self._f_sweep}")
        # m_info.append(f"{sep}\nx probe:\n{self._x_probe}")
        # m_info.append(f"{sep}\nf_probe:\n{self._f_probe}")
        # m_info.append(f"{sep}\np_probe:\n{self._p_probe}")
        # m_info.append(f"{sep}")
        # self.logger.info('\n'.join(m_info))

    @property
    def fr(self):
        return self._fr

    @property
    def _x2rx(self):
        """Model to create (r, x) from x."""
        return _x2rx(Qr=self._Qr, n_models=self._n_models)

    @property
    def _iq2rx(self):
        """Model to convert iq to rx in separate channels."""
        return ResonanceCircleInv(n_models=self._n_models)

    @property
    def _make_complex(self):
        """Model to combine real and imagine parts as complex values."""
        return ReadoutIQToComplex(n_models=self._n_models)

    @property
    def _complex_rx2iq(self):
        """Model to convert rx to iq in complex."""
        return ResonanceCircleComplex(n_models=self._n_models)

    @property
    def _complex_iq2rx(self):
        """Model to convert iq to rx in complex."""
        return ResonanceCircleComplexInv(n_models=self._n_models)

    @property
    def _p2x(self):
        """Model to evaluate x at p for given background and
        responsivity."""
        return OpticalDetune(
                background=self._background,
                responsivity=self._responsivity,
                n_models=self._n_models)

    @property
    def _x_sweep(self):
        """Model to evaluate IQ at x for given Qr."""
        return self._x2rx | self._make_complex | self._complex_rx2iq

    # @property
    # def _f_sweep(self):
    #     """Model to evaluate IQ at f for given fr and Qr."""
    #     return ResonanceCircleSweepComplex(fr=self._fr, Qr=self._Qr)

    # @property
    # def _f_probe(self):
    #     """Model to evaluate IQ at (fr, Qr) for given fp."""
    #     return ResonanceCircleProbeComplex()

    def sweep_x(self, n_steps=None, n_fwhms=None, xlim=None):
        """Return a resonance circle sweep."""
        if n_fwhms is None and xlim is None:
            raise ValueError("n_fwhms or xlim is required.")
        if xlim is None:
            xlim = (-0.5 * n_fwhms * self.fwhm_x,
                    0.5 * n_fwhms * self.fwhm_x)
        # get grid
        xs = np.linspace(*xlim, n_steps).T
        iqs = self._x_sweep(xs)
        return xs, iqs

    def probe_p(self, pwrs, fp=None, readout_model=None):
        """Return detector response for given optical power and probe
        frequency."""
        p2x = self._p2x
        x2rx = self._x2rx
        self.logger.debug(f"probe with n_mdoels={len(p2x)}")
        rs, xs = (p2x | x2rx)(pwrs)
        if fp is not None:
            broadcast_shape = [1, ] * pwrs.ndim
            broadcast_shape[p2x.model_set_axis] = len(p2x)
            xs_instru = InstrumentalDetune()(fp, self._fr).reshape(
                    tuple(broadcast_shape))
            xs = xs + xs_instru
        iqs = (self._make_complex | self._complex_rx2iq)(rs, xs)
        if readout_model is None:
            return rs, xs, iqs
        iqs_readout = readout_model(iqs, (xs + 1) * self._fr[:, np.newaxis])
        self.logger.debug(
                f'apply readout model '
                f'[{np.abs(iqs).min()}:{np.abs(iqs).max()}] -> '
                f'[{np.abs(iqs_readout).min()}:{np.abs(iqs_readout).max()}]'
                )
        return rs, xs, iqs_readout

    def solve_x(self, *args):
        """Return x for given detector response.
        """
        if len(args) == 1:
            # complex
            return self._complex_iq2rx(args[0])
        return self._iq2rx(*args)

    @property
    def fwhm_x(self):
        """Return the resonance FWHM in unit of x."""
        return 1. / self._Qr

    @property
    def fwhm_f(self):
        """Return the resonance FWHM in unit of Hz."""
        return self.fwhm_x * self._fr
