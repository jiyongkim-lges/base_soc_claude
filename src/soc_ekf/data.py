"""Loading the Samsung INR21700-30T dataset (Kollmeyer, McMaster, Mendeley 9xyvy2njj3 v2).

Sign convention used everywhere in this package: ``i`` is current in A, **discharge positive**.
(The raw files use discharge negative.)
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import scipy.io as sio

ROOT = pathlib.Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "samsung_30t"
TEMPS = (40, 25, 10, 0, -10, -20)
DRIVE_CYCLES = ("UDDS", "HWFET", "LA92", "US06") + tuple(f"Mixed{k}" for k in range(1, 9))


@dataclass
class Record:
    """One test file. Arrays are sorted by time with duplicate timestamps removed."""

    name: str
    temp: int  # chamber (ambient) temperature, degC
    t: np.ndarray  # s, starts at 0
    i: np.ndarray  # A, discharge positive
    v: np.ndarray  # V
    temp_cell: np.ndarray  # degC

    @property
    def ah_dis(self) -> np.ndarray:
        """Cumulative discharged charge in Ah.

        The Digatron logs each sample with the current of the interval that just ended, so the
        integral uses the right-hand value (this reproduces the tester's own Ah counter; trapezoidal
        integration drifts by ~0.2 Ah over the sparsely logged HPPC test)."""
        return cum_right(self.i, self.t) / 3600.0


def cum_right(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(y, dtype=float)
    out[1:] = np.cumsum(y[1:] * np.diff(x))
    return out


def find_file(temp: int, kind: str) -> pathlib.Path:
    folder = RAW / f"{temp}degC"
    hits = sorted(folder.glob(f"* *_{kind}_{temp}degC_IN21700_30T.mat"))
    if not hits:
        raise FileNotFoundError(f"no {kind} file at {temp} degC in {folder}")
    return hits[-1]  # last one if a test was restarted (e.g. -20degC Charge1)


def load(temp: int, kind: str) -> Record:
    path = find_file(temp, kind)
    m = sio.loadmat(path, squeeze_me=True, struct_as_record=False)["meas"]
    t = np.asarray(m.Time, dtype=float)
    order = np.argsort(t, kind="stable")
    t = t[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    sel = order[keep]
    t = t[keep]
    return Record(
        name=kind,
        temp=temp,
        t=t - t[0],
        i=-np.asarray(m.Current, dtype=float)[sel],
        v=np.asarray(m.Voltage, dtype=float)[sel],
        temp_cell=np.asarray(m.Battery_Temp_degC, dtype=float)[sel],
    )
