#! /usr/bin/env python

from ..types import KidsDataType
from ...wcs.frames import (NDDataFrame, )


_REQUIRES = ['astropy', 'gwcs']


__all__ = ["NDDataFrameType", ]


class NDDataFrameType(KidsDataType):

    name = "nddataframe"
    types = [NDDataFrame, ]
    version = '1.0.0'

    @classmethod
    def from_tree(cls, node, ctx):
        kwargs = {'name': node['name']}

        if 'naxes' in node:
            kwargs['naxes'] = node['naxes']

        if 'axes_order' in node:
            kwargs['axes_order'] = tuple(node['axes_order'])

        return NDDataFrame(**kwargs)

    @classmethod
    def to_tree(cls, frame, ctx):

        node = {}

        node['name'] = frame.name

        # We want to check that it is exactly this type and not a subclass
        if frame.axes_order == tuple(range(frame.naxes)):
            node['naxes'] = frame.naxes
        else:
            node['axes_order'] = list(frame.axes_order)
        return node

    @classmethod
    def assert_equal(cls, old, new):
        assert old.name == new.name
        assert old.axes_order == new.axes_order
