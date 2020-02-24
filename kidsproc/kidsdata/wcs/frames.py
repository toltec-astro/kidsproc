#! /usr/bin/env python


"""
This module defines the frames.

The implementation follows those in `gwcs.coorinate_frame`.
"""

from astropy import units as u


__all__ = ['NDDataFrame', ]


class NDDataFrame(object):
    """
    An n-dimensional index frame.

    This can be used to represent the indices of a n-dim array data.
    This is different from `gwcs.coordindate_frame`

    Parameters
    ----------
    axes_order : tuple of int
        A dimension in the input data that corresponds to this axis.

    name : str
        Name of this frame.
    """

    def __init__(
            self, naxes=None, axes_order=None, name=None):
        if naxes is None and axes_order is not None:
            naxes = len(axes_order)
        elif naxes is not None and axes_order is None:
            axes_order = range(naxes)
        else:
            raise ValueError("need to specify one of naxes or axes_order")
        self._naxes = naxes
        self._axes_order = tuple(axes_order)
        self._axes_names = tuple(f'{i}' for i in range(self._naxes))
        self._axes_type = ('INDEX', ) * self._naxes
        self._unit = (u.dimensionless_unscaled, ) * self._naxes
        self._axis_physical_types = tuple(
                f'custom:axis{i}' for i in range(self._naxes))
        if name is None:
            self._name = self.__class__.__name__
        else:
            self._name = name

    def __repr__(self):
        fmt = f'<{self.__class__.__name__}(name="{self.name}", ' \
            f'unit={self.unit}, axes_names={self.axes_names}, ' \
            f'axes_order={self.axes_order}'
        return fmt

    def __str__(self):
        if self._name is not None:
            return self._name
        return self.__class__.__name__

    @property
    def name(self):
        """ A custom name of this frame."""
        return self._name

    @name.setter
    def name(self, val):
        """ A custom name of this frame."""
        self._name = val

    @property
    def naxes(self):
        """ The number of axes in this frame."""
        return self._naxes

    @property
    def unit(self):
        """The unit of this frame."""
        return self._unit

    @property
    def axes_names(self):
        """ Names of axes in the frame."""
        return self._axes_names

    @property
    def axes_order(self):
        """ A tuple of indices which map inputs to axes."""
        return self._axes_order

    @property
    def axes_type(self):
        """ Type of this frame : 'SPATIAL', 'SPECTRAL', 'TIME'. """
        return self._axes_type

    def coordinates(self, *args):
        """ Create world coordinates object"""
        args = [args[i] for i in self.axes_order]
        coo = tuple(
                arg * un if not hasattr(arg, "to") else arg.to(un)
                for arg, un in zip(args, self.unit))
        return coo

    def coordinate_to_quantity(self, *coords):
        """
        Given a rich coordinate object return an astropy quantity object.
        """
        # NoOp leaves it to the model to handle
        return coords

    @property
    def axis_physical_types(self):
        return self._axis_physical_types
