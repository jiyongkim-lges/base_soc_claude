"""Lookup tables over (SOC, temperature) with bilinear interpolation and edge clamping."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

ECM_SOC = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0])
ECM_FIELDS = ("r0", "r1", "tau1", "r2", "tau2")


@dataclass
class Table2D:
    soc: np.ndarray  # ascending
    temps: np.ndarray  # ascending, degC
    values: dict[str, np.ndarray]  # name -> array (len(soc), len(temps))

    def __post_init__(self):
        self._f = {
            k: RegularGridInterpolator((self.soc, self.temps), v, bounds_error=False, fill_value=None)
            for k, v in self.values.items()
        }

    def at(self, name: str, soc, temp):
        soc = np.clip(np.asarray(soc, dtype=float), self.soc[0], self.soc[-1])
        temp = np.clip(np.asarray(temp, dtype=float), self.temps[0], self.temps[-1])
        soc, temp = np.broadcast_arrays(soc, temp)
        return self._f[name](np.stack([soc.ravel(), temp.ravel()], axis=-1)).reshape(soc.shape)

    def to_long(self) -> pd.DataFrame:
        rows = []
        for j, T in enumerate(self.temps):
            for k, s in enumerate(self.soc):
                rows.append({"temp": T, "soc": s, **{n: v[k, j] for n, v in self.values.items()}})
        return pd.DataFrame(rows)

    @classmethod
    def from_long(cls, df: pd.DataFrame) -> "Table2D":
        soc = np.sort(df.soc.unique())
        temps = np.sort(df.temp.unique())
        names = [c for c in df.columns if c not in ("soc", "temp")]
        values = {n: df.pivot(index="soc", columns="temp", values=n).loc[soc, temps].to_numpy() for n in names}
        return cls(soc, temps, values)


def ecm_table(pulses: pd.DataFrame, current: float, soc_grid=ECM_SOC) -> Table2D:
    """2RC table from discharge pulses of the given current (A). Fits that hit a bound or whose RMSE
    is > 3x the median at that temperature are dropped; gaps are filled by interpolation in SOC with
    edge hold."""
    d = pulses[(pulses.current - current).abs() < 0.25 * current].copy()
    temps = np.sort(d.temp.unique())
    values = {n: np.zeros((len(soc_grid), len(temps))) for n in ECM_FIELDS}
    for j, T in enumerate(temps):
        dt = d[(d.temp == T) & (d.soc <= 1.0 + 1e-3)].sort_values("soc")
        ok = (
            (dt.fit_rmse_mv <= 3 * dt.fit_rmse_mv.median())
            & dt.tau1.between(0.11, 9.9)
            & dt.tau2.between(10.5, 1990)
        )
        dt = dt[ok]
        for n in ECM_FIELDS:
            col = "r0" if n == "r0" else n
            values[n][:, j] = np.interp(soc_grid, dt.soc, dt[col])
    return Table2D(np.asarray(soc_grid, float), temps.astype(float), values)


def ocv_table(soc_grid, table: dict[int, np.ndarray]) -> Table2D:
    temps = np.array(sorted(table))
    return Table2D(np.asarray(soc_grid, float), temps.astype(float), {"ocv": np.stack([table[T] for T in temps], axis=1)})


I_GRID = np.array([1.5, 3.0, 6.0, 12.0, 18.0, 24.0, 36.0])  # A, |current| axis of the 3D table


@dataclass
class EcmTable3D:
    """2RC parameters over (SOC, temperature, |current|).

    At low temperature the HPPC polarisation resistance falls strongly with pulse current
    (Butler-Volmer behaviour); a single-current table over-predicts voltage swings in drive cycles.
    The current axis is interpolated in log(|I|) and clamped to ``I_GRID``."""

    soc: np.ndarray
    temps: np.ndarray
    i_grid: np.ndarray
    values: dict[str, np.ndarray]  # name -> (len(soc), len(temps), len(i_grid))

    def __post_init__(self):
        axes = (self.soc, self.temps, np.log(self.i_grid))
        self._f = {k: RegularGridInterpolator(axes, v, bounds_error=False, fill_value=None) for k, v in self.values.items()}

    def at(self, name: str, soc, temp, current):
        soc = np.clip(np.asarray(soc, dtype=float), self.soc[0], self.soc[-1])
        temp = np.clip(np.asarray(temp, dtype=float), self.temps[0], self.temps[-1])
        li = np.log(np.clip(np.abs(np.asarray(current, dtype=float)), self.i_grid[0], self.i_grid[-1]))
        soc, temp, li = np.broadcast_arrays(soc, temp, li)
        pts = np.stack([soc.ravel(), temp.ravel(), li.ravel()], axis=-1)
        return self._f[name](pts).reshape(soc.shape)

    def to_long(self) -> pd.DataFrame:
        rows = []
        for j, T in enumerate(self.temps):
            for m, I in enumerate(self.i_grid):
                for k, s in enumerate(self.soc):
                    rows.append({"temp": T, "current": I, "soc": s, **{n: v[k, j, m] for n, v in self.values.items()}})
        return pd.DataFrame(rows)

    @classmethod
    def from_long(cls, df: pd.DataFrame) -> "EcmTable3D":
        soc, temps, ig = (np.sort(df[c].unique()) for c in ("soc", "temp", "current"))
        d = df.set_index(["soc", "temp", "current"]).sort_index()
        values = {n: d[n].to_numpy().reshape(len(soc), len(temps), len(ig)) for n in ECM_FIELDS}
        return cls(soc, temps, ig, values)


def ecm_table_3d(pulses: pd.DataFrame, i_grid=I_GRID, soc_grid=ECM_SOC, min_points=5) -> EcmTable3D:
    """Per temperature, build an SOC table for every pulse current with >= ``min_points`` pulses,
    then interpolate across currents in log(|I|) with edge hold."""
    d = pulses[pulses.current > 0]
    temps = np.sort(d.temp.unique()).astype(float)
    values = {n: np.zeros((len(soc_grid), len(temps), len(i_grid))) for n in ECM_FIELDS}
    for j, T in enumerate(temps):
        dt = d[d.temp == T]
        levels = sorted({float(np.round(c)) for c in dt.current})
        levels = [L for L in levels if ((dt.current - L).abs() < 0.25 * L).sum() >= min_points]
        per = {L: ecm_table(dt, L, soc_grid).values for L in levels}
        for n in ECM_FIELDS:
            arr = np.stack([per[L][n][:, 0] for L in levels], axis=1)
            for k in range(len(soc_grid)):
                values[n][k, j, :] = np.interp(np.log(i_grid), np.log(levels), arr[k])
    return EcmTable3D(np.asarray(soc_grid, float), temps, np.asarray(i_grid, float), values)
