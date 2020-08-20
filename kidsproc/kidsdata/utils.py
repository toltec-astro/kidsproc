#! /usr/bin/env python

from astropy.nddata import NDDataRef


__all__ = ['ExtendedNDDataRef', ]


class ExtendedNDDataRef(NDDataRef):
    """A little bit more than `~astropy.nddata.NDDataRef`."""

    def __repr__(self):
        data = getattr(self, '_data', None)
        shape = data.shape if data is not None else '(empty)'
        return f"{self.__class__.__name__}{shape}"

    def __getitem__(self, item):
        """Implement slice of additional attributes along with this object.

        Define the list of attributes to slice in `_attr_to_slice` class
        attributes.

        """
        inst = super().__getitem__(item)
        inst.__dict__.update(self._slice_extra(item))
        return inst


class FrequencyDivisionMultiplexingDataRef(ExtendedNDDataRef):
    """A generic container for frequency division multiplexed data.

    The FDM technique is used in reading the data from large format KIDs
    array. This class defines a generic structure to handle multi-dimensional
    data generated this way.

    The data shall be organized such that the first axis is for
    the different readout channels known as ``tones``.

    Parameters
    ----------
    tones : array-like
        The tone properties.
    data : `astropy.nddata.NDData`
        The data.
    kwargs :
        keyword arguments to pass to `astropy.nddata.NDDataRef`.
    """

    def _slice_extra(self, item):
        if not isinstance(item, tuple):
            item = (item, )
        return {
                '_tones': self._tones[item[0]]
                }

    def __init__(
            self, tones=None, data=None, **kwargs):

        # check lengths of axis
        if data is not None and tones is not None:
            if data.shape[0] != len(tones):
                raise ValueError(
                    f"data of shape {data.shape} is incompatible with the"
                    f" length of the tones {len(tones)}.")

        self._tones = tones
        super().__init__(data=data, **kwargs)

    @property
    def tones(self):
        """The tones."""
        return self._tones
