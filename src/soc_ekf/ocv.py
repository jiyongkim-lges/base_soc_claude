"""OCV-SOC curves from the C/20 discharge/charge test.

SOC is defined against a single reference capacity ``Q_REF`` (C/20 discharge capacity at 25 degC),
so the same SOC means the same amount of stored charge at every temperature (see docs/adr/0001).
"""
from __future__ import annotations

import numpy as np

from .data import Record, load

SOC_GRID = np.round(np.linspace(0.0, 1.0, 201), 4)


def c20_capacity(rec: Record) -> float:
    """Ah removed during the C/20 discharge part."""
    ah = rec.ah_dis
    dis = rec.i > 0.05
    return float(ah[dis][-1] - ah[dis][0])


def c20_branches(rec: Record, q_ref: float) -> tuple[np.ndarray, np.ndarray]:
    """C/20 discharge and charge voltage on ``SOC_GRID`` (NaN outside the measured window)."""
    ah = rec.ah_dis
    dis = rec.i > 0.05
    ch = rec.i < -0.05
    q_dis = ah[dis][-1] - ah[dis][0]
    soc_dis = 1.0 - (ah[dis] - ah[dis][0]) / q_ref
    soc_end = 1.0 - q_dis / q_ref
    # Charge branch: starts at soc_end, scaled so that it ends at SOC = 1 (coulombic efficiency).
    q_ch_ah = -(ah[ch] - ah[ch][0])
    soc_ch = soc_end + q_ch_ah / q_ch_ah[-1] * (1.0 - soc_end)
    # Drop the CV tail of charge (same SOC, falling current) so the branch is monotonic.
    i_ch = -rec.i[ch]
    cc = i_ch > 0.9 * np.median(i_ch)
    soc_ch, v_ch = soc_ch[cc], rec.v[ch][cc]
    keep = np.concatenate([[True], np.diff(soc_ch) > 1e-6])
    soc_ch, v_ch = soc_ch[keep], v_ch[keep]

    vd = np.interp(SOC_GRID, soc_dis[::-1], rec.v[dis][::-1], left=np.nan, right=np.nan)
    vc = np.interp(SOC_GRID, soc_ch, v_ch, left=np.nan, right=np.nan)
    return vd, vc


def ocv_from_c20(rec: Record, q_ref: float, kind: str = "avg", r_dis: float = 0.0) -> np.ndarray:
    """OCV on ``SOC_GRID`` (NaN outside the measured window).

    kind="avg": mean of the C/20 discharge and charge branches (cancels IR drop, averages hysteresis).
    kind="dis": discharge branch with its IR drop added back (``r_dis`` * C/20 current); this is the
    hysteresis branch the cell sits on during discharge-dominated drive cycles."""
    vd, vc = c20_branches(rec, q_ref)
    if kind == "avg":
        return 0.5 * (vd + vc)
    if kind == "dis":
        i_c20 = float(np.median(rec.i[rec.i > 0.05]))
        return vd + r_dis * i_c20
    raise ValueError(kind)


def build_ocv_table(temps, kind: str = "avg", r_dis: dict | None = None) -> tuple[float, dict[int, np.ndarray]]:
    """Q_ref and gap-free OCV curves on ``SOC_GRID`` per temperature.

    The 25 degC curve is completed by linear extrapolation of its end slopes; other temperatures
    borrow the 25 degC shape outside their measured SOC window (see ``fill_edges``)."""
    q_ref = c20_capacity(load(25, "C20DisCh"))
    r_dis = r_dis or {}
    raw = {T: ocv_from_c20(load(T, "C20DisCh"), q_ref, kind, r_dis.get(T, 0.0)) for T in temps}
    ref = extrapolate_linear(raw[25])
    table = {T: (ref if T == 25 else fill_edges(raw[T], ref)) for T in temps}
    # keep OCV monotonic in SOC (noise at the flat top can produce tiny inversions)
    table = {T: np.maximum.accumulate(v) for T, v in table.items()}
    return q_ref, table


def extrapolate_linear(v: np.ndarray, n_fit: int = 4) -> np.ndarray:
    v = v.copy()
    ok = np.flatnonzero(~np.isnan(v))
    lo, hi = ok[0], ok[-1]
    x = SOC_GRID
    if lo > 0:
        a, b = np.polyfit(x[lo : lo + n_fit], v[lo : lo + n_fit], 1)
        v[:lo] = a * x[:lo] + b
    if hi < len(v) - 1:
        a, b = np.polyfit(x[hi - n_fit + 1 : hi + 1], v[hi - n_fit + 1 : hi + 1], 1)
        v[hi + 1 :] = a * x[hi + 1 :] + b
    return v


def fill_edges(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Fill NaNs at the SOC ends by following the shape of ``ref`` (the 25 degC curve),
    offset to join the measured part continuously."""
    v = v.copy()
    ok = np.flatnonzero(~np.isnan(v))
    lo, hi = ok[0], ok[-1]
    v[:lo] = ref[:lo] + (v[lo] - ref[lo])
    v[hi + 1 :] = ref[hi + 1 :] + (v[hi] - ref[hi])
    return v
