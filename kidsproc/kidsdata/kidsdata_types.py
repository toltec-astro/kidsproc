# -*- coding: utf-8 -*-

"""
Defines a ``KidsDataType`` used by `kidsutils.kidsdata`.

All types are added automatically to ``_kidsdata_types`` and the
KidsDataExtension.

"""

from asdf.types import ExtensionTypeMeta, CustomType


__all__ = ['KidsDataType', ]


_kidsdata_types = set()


class KidsDataTypeMeta(ExtensionTypeMeta):
    """
    Keeps track of `KidsDataType` subclasses that are created so that they can
    be stored automatically by astropy extensions for ASDF.
    """
    def __new__(mcls, name, bases, attrs):
        cls = super().__new__(mcls, name, bases, attrs)
        # Classes using this metaclass are automatically added to the list of
        # jwst types and JWSTExtensions.types.
        if issubclass(cls, KidsDataType):
            _kidsdata_types.add(cls)
        return cls


class KidsDataType(CustomType, metaclass=ExtensionTypeMeta):
    """
    This class represents types that have schemas and tags
    implemented within `tolteca.kidsutils.kidsdata`.

    """
    organization = 'umass.edu'
    standard = 'toltec'

    def __init_subclass__(cls):
        super().__init_subclass__()
        _kidsdata_types.add(cls)
