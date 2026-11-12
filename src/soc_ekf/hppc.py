"""HPPC pulse detection and parameter extraction.

Each SOC block of the 30T HPPC test: 10 s discharge pulse -> rest -> 10 s charge pulse -> rest ->
small 0.3C rebalance, repeated for 1, 2, 6, 12C (reduced at low temperature); then a 0.3C discharge
to the next SOC point and a long rest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import Record
from .ecm import fit_pulse

ON = 0.3  # A, threshold for "current flowing"


@dataclass
class Pulse:
    k0: int  # first sample with current
    k1: int  # last sample with current
    k_end: int  # last sample of the following rest (before next current)
    current: float  # A, discharge positive
    duration: float  # s
    rest_after: float  # s


def find_pulses(rec: Record, min_dur=9.0, max_dur=11.0) -> list[Pulse]:
    on = np.abs(rec.i) > ON
    edges = np.flatnonzero(np.diff(on.astype(int)))
    starts = edges[on[edges + 1]] + 1
    ends = edges[~on[edges + 1]]
    pulses = []
    for k0 in starts:
        later = ends[ends >= k0]
        if len(later) == 0:
            break
        k1 = int(later[0])
        nxt = starts[starts > k1]
        k_end = int(nxt[0] - 1) if len(nxt) else len(rec.t) - 1
        dur = rec.t[k1] - rec.t[k0]
        if min_dur <= dur <= max_dur and k0 > 0 and not on[k0 - 1]:
            pulses.append(
                Pulse(int(k0), k1, k_end, float(np.median(rec.i[k0 : k1 + 1])), float(dur), float(rec.t[k_end] - rec.t[k1]))
            )
    return pulses


def extract(rec: Record, q_ref: float, ocv_of_soc, max_rest=600.0) -> list[dict]:
    """One row per 10 s pulse: SOC, current, R0, R_1s, R_10s, and (discharge pulses) 2RC fit.

    ``ocv_of_soc`` maps SOC array -> OCV at this temperature (used for SOC drift in the fit window).
    SOC is from Coulomb counting from the start of the file (fully charged)."""
    ah = rec.ah_dis
    t, i, v = rec.t, rec.i, rec.v
    rows = []
    pulses = find_pulses(rec)
    dis_starts = np.array([p.k0 for p in pulses if p.current > 0] + [len(t)])
    for p in pulses:
        kp = p.k0 - 1  # last rest sample before the pulse
        soc0 = 1.0 - ah[kp] / q_ref
        di = i[p.k0] - i[kp]
        # R0: instantaneous step (first sample after the edge, ~0.1 s)
        r0 = (v[kp] - v[p.k0]) / di
        # R_1s and R_10s: voltage change 1 s after pulse start and at pulse end (~9.9 s)
        k1s = np.searchsorted(t, t[p.k0] + 1.0)
        r1s = (v[kp] - v[k1s]) / di
        r10s = (v[kp] - v[p.k1]) / di
        row = dict(
            soc=soc0,
            current=p.current,
            c_rate=p.current / 3.0,
            r0=r0,
            r_1s=r1s,
            r_10s=r10s,
            t_10s=t[p.k1] - t[p.k0],
            temp_cell=float(np.mean(rec.temp_cell[p.k0 : p.k1 + 1])),
        )
        if p.current > 0:
            # 2RC fit on [pre-pulse rest, pulse, everything that follows (rests, charge/CV steps,
            # rebalance) up to the next discharge pulse, at most ``max_rest`` s after the pulse]
            k_next = int(dis_starts[dis_starts > p.k0][0]) - 1
            k_stop = min(k_next, int(np.searchsorted(t, t[p.k1] + max_rest)), len(t) - 1)
            sl = slice(kp, k_stop + 1)
            tw = t[sl] - t[kp]
            iw = i[sl]
            soc_w = 1.0 - ah[sl] / q_ref
            ocv_w = ocv_of_soc(soc_w)
            ocv_w = ocv_w - ocv_w[0] + v[kp]  # anchor to the measured rested voltage
            fit = fit_pulse(tw, iw, v[sl], ocv_w, r0)
            row.update(
                r0_fit=fit.params.r0,
                r1=fit.params.r1,
                tau1=fit.params.tau1,
                r2=fit.params.r2,
                tau2=fit.params.tau2,
                fit_rmse_mv=fit.rmse * 1e3,
                fit_window_s=float(tw[-1]),
            )
        rows.append(row)
    return rows


def rest_ocv_points(rec: Record, q_ref: float, min_rest=1800.0) -> list[tuple[float, float]]:
    """(SOC, V) at the end of each long rest (>= ``min_rest`` s) -- an independent OCV check."""
    on = np.abs(rec.i) > 0.01
    pts = []
    k = 0
    n = len(on)
    ah = rec.ah_dis
    while k < n:
        if on[k]:
            k += 1
            continue
        j = k
        while j < n and not on[j]:
            j += 1
        if rec.t[j - 1] - rec.t[k] >= min_rest:
            pts.append((1.0 - ah[j - 1] / q_ref, float(rec.v[j - 1])))
        k = j
    return pts
