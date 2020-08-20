#! /usr/bin/env python

from asdf.types import ExtensionTypeMeta, CustomType


__all__ = ['KidsDataType', ]


_kidsdata_types = set()


class KidsDataTypeMeta(ExtensionTypeMeta):
    """
    Keeps track of `KidsDataType` subclasses that are created so that they can
    be stored automatically by `kidsproc.kidsdata` extensions for ASDF.
    """
    def __new__(mcls, name, bases, attrs):
        cls = super().__new__(mcls, name, bases, attrs)
        # Classes using this metaclass are automatically added to the list of
        # kidsdata types and KidsDataExtensions.types.
        if issubclass(cls, KidsDataType):
            _kidsdata_types.add(cls)
        return cls


class KidsDataType(CustomType, metaclass=ExtensionTypeMeta):
    """
    This class represents types that have schemas and tags
    implemented within `kidsproc.kidsdata`.

    """
    organization = 'toltecdr.astro.umass.edu'
    standard = 'kidsdata'

    def __init_subclass__(cls):
        super().__init_subclass__()
        # store subclasses in _kidsdata_types
        _kidsdata_types.add(cls)
