# Licensed under a 3-clause BSD style license - see LICENSE.rst
# -*- coding: utf-8 -*-
from ...kidsdata_types import KidsDataType
from ..frames import (NDDataFrame, )


_REQUIRES = ['astropy', 'gwcs']


__all__ = ["NDDataFrameType", ]


class NDDataFrameType(KidsDataType):
    name = "nddataframe"
    requires = _REQUIRES
    types = [NDDataFrame, ]
    version = '1.0.0'

    @classmethod
    def _from_tree(cls, node, ctx):
        kwargs = {'name': node['name']}

        if 'naxes' in node:
            kwargs['naxes'] = node['naxes']

        if 'axes_order' in node:
            kwargs['axes_order'] = tuple(node['axes_order'])

        return kwargs

    @classmethod
    def _to_tree(cls, frame, ctx):

        node = {}

        node['name'] = frame.name

        # We want to check that it is exactly this type and not a subclass
        if frame.axes_order == tuple(range(frame.naxes)):
            node['naxes'] = frame.naxes
        else:
            node['axes_order'] = list(frame.axes_order)
        return node

    @classmethod
    def _assert_equal(cls, old, new):
        assert old.name == new.name  # nosec
        assert old.axes_order == new.axes_order  # nosec

    @classmethod
    def assert_equal(cls, old, new):
        cls._assert_equal(old, new)

    @classmethod
    def from_tree(cls, node, ctx):
        node = cls._from_tree(node, ctx)
        return NDDataFrame(**node)

    @classmethod
    def to_tree(cls, frame, ctx):
        return cls._to_tree(frame, ctx)
