#! /usr/bin/env python

from pathlib import Path
from asdf import util
from asdf.extension import BuiltinExtension

from .wcs.tags import *  # noqa: F403, F401
from .kidsdata_types import _kidsdata_types


SCHEMA_PATH = Path(__file__).with_name('schemas').resolve()
ORG_URL_BASE = 'astro.umass.edu'
STD_STR = 'kidsdata'
SCM_URL = f'http://{ORG_URL_BASE}/schemas/{STD_STR}'


class KidsDataExtension(BuiltinExtension):

    @property
    def types(self):
        return _kidsdata_types

    @property
    def tag_mapping(self):
        return [(
            f'tag:{ORG_URL_BASE}:{STD_STR}',
            f'http://{SCM_URL}{{tag_suffix}}')]

    @property
    def url_mapping(self):
        return [(
            SCM_URL,
            util.filepath_to_url(
                SCHEMA_PATH.joinpath(ORG_URL_BASE).joinpath(
                    f"{STD_STR}{{url_suffix}}.yaml").as_posix()))]
