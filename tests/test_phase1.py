import numpy as np

from soc_ekf.data import Record
from soc_ekf.ecm import RC2, fit_pulse, simulate
from soc_ekf.hppc import extract, find_pulses


def synthetic_pulse(p: RC2, amp=6.0, dt=0.1, rest_before=5.0, dur=9.9, rest_after=150.0, ocv0=3.8):
    t = np.arange(0.0, rest_before + dur + rest_after, dt)
    i = np.where((t >= rest_before) & (t < rest_before + dur), amp, 0.0)
    v = simulate(t, i, np.full_like(t, ocv0), p)
    return t, i, v


def test_fit_pulse_recovers_known_parameters():
    true = RC2(r0=0.020, r1=0.008, tau1=1.5, r2=0.012, tau2=60.0)
    t, i, v = synthetic_pulse(true)
    fit = fit_pulse(t, i, v, np.full_like(t, 3.8), r0_guess=0.02)
    got = fit.params
    assert fit.rmse < 1e-5
    for name in ("r0", "r1", "tau1", "r2", "tau2"):
        assert abs(getattr(got, name) / getattr(true, name) - 1) < 0.02, name


def test_extract_finds_pulse_and_resistances():
    true = RC2(r0=0.020, r1=0.008, tau1=1.5, r2=0.012, tau2=60.0)
    t, i, v = synthetic_pulse(true, amp=6.0)
    rec = Record("HPPC", 25, t, i, v, np.full_like(t, 25.0))
    pulses = find_pulses(rec)
    assert len(pulses) == 1 and abs(pulses[0].current - 6.0) < 1e-9
    (row,) = extract(rec, q_ref=3.0, ocv_of_soc=lambda s: np.full_like(s, 3.8))
    # step is sampled 0.1 s after the edge (ZOH), so R0 ~= true R0 + small RC contribution
    assert abs(row["r0"] - 0.020) < 1e-3
    assert row["r_1s"] > row["r0"] and row["r_10s"] > row["r_1s"]
    assert abs(row["r1"] / 0.008 - 1) < 0.02 and abs(row["tau2"] / 60 - 1) < 0.02


def test_ecm_table_3d_roundtrip_and_current_axis():
    import pandas as pd

    from soc_ekf.tables import EcmTable3D, ecm_table_3d

    rows = []
    for T in (0, 25):
        for I, scale in ((3.0, 2.0), (12.0, 1.0)):  # resistance halves from 3 A to 12 A
            for s in np.linspace(0.1, 1.0, 6):
                rows.append(dict(temp=T, soc=s, current=I, r0=0.01 * scale, r1=0.005 * scale, tau1=1.0,
                                 r2=0.01 * scale, tau2=50.0, fit_rmse_mv=1.0))
    tab = ecm_table_3d(pd.DataFrame(rows))
    assert np.isclose(tab.at("r0", 0.5, 10, 3.0), 0.02)
    assert np.isclose(tab.at("r0", 0.5, 10, 12.0), 0.01)
    assert np.isclose(tab.at("r0", 0.5, 10, 6.0), 0.015)  # midway in log(I)
    assert np.isclose(tab.at("r0", 0.5, 10, -12.0), 0.01)  # |I| for charge
    back = EcmTable3D.from_long(tab.to_long())
    assert np.isclose(back.at("r2", 0.3, 0, 5.0), tab.at("r2", 0.3, 0, 5.0))
