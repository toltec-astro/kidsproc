#! /usr/bin/env python

import pytest

# Make sure these tests do not run if ASDF is not installed
pytest.importorskip('asdf')

from asdf.tests.helpers import assert_roundtrip_tree  # noqa: E402

from kidsproc.kidsdata.wcs.frames import NDDataFrame  # noqa: E402


def test_nddataframe(tmpdir):

    frame = NDDataFrame(name='test_frame', naxes=2)

    tree = dict(frame=frame)
    assert_roundtrip_tree(tree, tmpdir)

    frame = NDDataFrame(name='test_frame', axes_order=[1, 0])

    tree = dict(frame=frame)
    assert_roundtrip_tree(tree, tmpdir)
