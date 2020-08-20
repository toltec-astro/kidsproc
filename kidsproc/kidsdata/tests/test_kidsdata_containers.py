#! /usr/bin/env python


import numpy as np
from .. import MultiSweep, Sweep, TimeStream
import astropy.units as u
from astropy.tests.helper import assert_quantity_allclose


def test_sweep():

    fs = np.arange(10) << u.Hz
    S21 = np.arange(10, 20) + 1.j * np.arange(10, 20)
    D21 = np.arange(10, 20) * u.adu / u.Hz

    swp = Sweep(S21=S21, frequency=fs, D21=D21)

    assert_quantity_allclose(
            swp.frequency,
            fs)

    assert_quantity_allclose(
            swp.S21,
            S21 << u.adu
            )
    assert_quantity_allclose(
            swp.D21,
            D21
            )


def test_sweep_slice():

    fs = np.arange(10) << u.Hz
    S21 = np.arange(10, 20) + 1.j * np.arange(10, 20)
    D21 = np.arange(10, 20) * u.adu / u.Hz

    swp = Sweep(S21=S21, frequency=fs, D21=D21)

    s = slice(1, -1)

    swp = swp[s]

    assert_quantity_allclose(
            swp.frequency,
            fs[s])

    assert_quantity_allclose(
            swp.S21,
            S21[s] << u.adu
            )
    assert_quantity_allclose(
            swp.D21,
            D21[s]
            )

    swp = Sweep(S21=S21, frequency=fs, D21=D21)
    s = 0
    swp = swp[s]

    assert_quantity_allclose(
            swp.frequency,
            fs[s])

    assert_quantity_allclose(
            swp.S21,
            S21[s] << u.adu
            )
    assert_quantity_allclose(
            swp.D21,
            D21[s]
            )

    # mask
    swp = Sweep(S21=S21, frequency=fs, D21=D21)
    s = fs > 5 << u.Hz
    swp = swp[s]

    assert_quantity_allclose(
            swp.frequency,
            fs[s])

    assert_quantity_allclose(
            swp.S21,
            S21[s] << u.adu
            )
    assert_quantity_allclose(
            swp.D21,
            D21[s]
            )


def test_multi_sweep():

    tones = np.arange(10) << u.Hz
    sweeps = np.arange(5) << u.Hz
    S21 = np.ones((len(tones), len(sweeps)), dtype=complex)

    swp = MultiSweep(tones=tones, sweeps=sweeps, S21=S21)

    assert_quantity_allclose(
            swp.tones,
            tones)

    assert_quantity_allclose(
            swp.sweeps,
            sweeps)

    assert_quantity_allclose(
            swp.frequency,
            np.sum(np.mgrid[:10, :5], axis=0) << u.Hz)

    assert_quantity_allclose(
            swp.S21,
            S21 << u.adu
            )


def test_multi_sweep_slice():

    tones = np.arange(10) << u.Hz
    sweeps = np.arange(5) << u.Hz
    S21 = np.ones((len(tones), len(sweeps)), dtype=complex)

    swp = MultiSweep(tones=tones, sweeps=sweeps, S21=S21)
    s = slice(1, -1)
    swp = swp[s]

    assert_quantity_allclose(
            swp.tones,
            tones[s])

    assert_quantity_allclose(
            swp.sweeps,
            sweeps)

    assert_quantity_allclose(
            swp.frequency,
            np.sum(np.mgrid[1:9, :5], axis=0) << u.Hz)

    assert_quantity_allclose(
            swp.S21,
            (S21 << u.adu)[s]
            )

    # slice two dims
    swp = MultiSweep(tones=tones, sweeps=sweeps, S21=S21)
    s = slice(1, -1)
    swp = swp[s, s]

    assert_quantity_allclose(
            swp.tones,
            tones[s])

    assert_quantity_allclose(
            swp.sweeps,
            sweeps[s])

    assert_quantity_allclose(
            swp.frequency,
            np.sum(np.mgrid[1:9, 1:4], axis=0) << u.Hz)

    assert_quantity_allclose(
            swp.S21,
            (S21 << u.adu)[s, s]
            )

    # slice second dim
    swp = MultiSweep(tones=tones, sweeps=sweeps, S21=S21)
    s = slice(1, -1)
    swp = swp[:, s]

    assert_quantity_allclose(
            swp.tones,
            tones)

    assert_quantity_allclose(
            swp.sweeps,
            sweeps[s])

    assert_quantity_allclose(
            swp.frequency,
            np.sum(np.mgrid[0:10, 1:4], axis=0) << u.Hz)

    assert_quantity_allclose(
            swp.S21,
            (S21 << u.adu)[:, s]
            )

    # extract tone
    swp = MultiSweep(tones=tones, sweeps=sweeps, S21=S21)
    s = 0
    swp = swp[s, :]

    assert_quantity_allclose(
            swp.tones,
            tones[s])

    assert_quantity_allclose(
            swp.sweeps,
            sweeps)

    assert_quantity_allclose(
            swp.frequency,
            np.sum(np.mgrid[0:1, 0:5], axis=0).ravel() << u.Hz)

    assert_quantity_allclose(
            swp.S21,
            (S21 << u.adu)[s, :]
            )


def test_timestream():

    tones = np.arange(10) << u.Hz
    times = np.arange(100) << u.s
    I = np.ones((len(tones), len(times)), dtype=float)  # noqa: E741
    Q = np.ones((len(tones), len(times)), dtype=float)
    r = np.ones((len(tones), len(times)), dtype=float)
    x = np.ones((len(tones), len(times)), dtype=float)

    ts = TimeStream(tones=tones, I=I, Q=Q, r=r, x=x)  # noqa: E741

    assert_quantity_allclose(
            ts.tones,
            tones)

    assert ts.times is None  # we did not pass the times there

    assert_quantity_allclose(
            ts.I,
            I << u.adu)

    assert_quantity_allclose(
            ts.Q,
            Q << u.adu)

    assert_quantity_allclose(
            ts.r,
            r)

    assert_quantity_allclose(
            ts.x,
            x)


def test_timestream_slice():

    tones = np.arange(10) << u.Hz
    times = np.arange(100) << u.s
    I = np.ones((len(tones), len(times)), dtype=float)  # noqa: E741
    Q = np.ones((len(tones), len(times)), dtype=float)
    r = np.ones((len(tones), len(times)), dtype=float)
    x = np.ones((len(tones), len(times)), dtype=float)

    ts = TimeStream(tones=tones, times=times, I=I, Q=Q, r=r, x=x)  # noqa: E741

    s = slice(1, -1)
    t1 = ts[s]

    assert_quantity_allclose(
            t1.tones,
            tones[s])

    assert_quantity_allclose(
            t1.times,
            times)

    assert_quantity_allclose(
            t1.I,
            (I << u.adu)[s]
            )

    s1 = slice(None, 2, 2)
    t1 = ts[s, s1]

    assert_quantity_allclose(
            t1.tones,
            tones[s])

    assert_quantity_allclose(
            t1.times,
            times[s1])

    assert_quantity_allclose(
            t1.I,
            (I << u.adu)[s, s1]
            )
