"""Phase 1: OCV, HPPC parameters (2RC, R_1s, R_10s), true SOC for drive cycles, model validation.

Outputs: params/*.csv, data/processed/*.npz, reports/01_dataset.md, reports/figures/*.png
Run: .venv/bin/python scripts/run_phase1.py
"""
from __future__ import annotations

import itertools
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from soc_ekf.data import DRIVE_CYCLES, ROOT, TEMPS, load
from soc_ekf.ecm import simulate_tables
from soc_ekf.hppc import extract, rest_ocv_points
from soc_ekf.ocv import SOC_GRID, build_ocv_table
from soc_ekf.tables import ecm_table, ecm_table_3d, ocv_table

PARAMS = ROOT / "params"
PROC = ROOT / "data" / "processed"
REP = ROOT / "reports"
FIG = REP / "figures"
VALIDATE_CYCLES = ("UDDS", "HWFET", "LA92", "US06")
PULSE_CHOICES = {"1C": 3.0, "2C": 6.0}


def main():
    for d in (PARAMS, PROC, FIG):
        d.mkdir(parents=True, exist_ok=True)

    # --- OCV (average branch) and HPPC extraction -------------------------------------------
    q_ref, ocv_avg = build_ocv_table(TEMPS, "avg")
    rows, rest_pts = [], []
    for T in TEMPS:
        rec = load(T, "HPPC")
        f = lambda s, T=T: np.interp(s, SOC_GRID, ocv_avg[T])
        for r in extract(rec, q_ref, f):
            rows.append({"temp": T, **r})
        rest_pts += [(T, s, v) for s, v in rest_ocv_points(rec, q_ref) if s <= 1.0]
    pulses = pd.DataFrame(rows)
    pulses.to_csv(PARAMS / "hppc_pulses.csv", index=False, float_format="%.6g")

    # C/20 polarisation for the discharge-branch OCV: median 1C R_10s per temperature
    r_dis = pulses[(pulses.current.round() == 3)].groupby("temp").r_10s.median().to_dict()
    _, ocv_dis = build_ocv_table(TEMPS, "dis", r_dis)
    ocv_tabs = {"avg": ocv_table(SOC_GRID, ocv_avg), "dis": ocv_table(SOC_GRID, ocv_dis)}
    for k, tab in ocv_tabs.items():
        df = tab.to_long().pivot(index="soc", columns="temp", values="ocv")
        df.to_csv(PARAMS / f"ocv_soc_{k}.csv", float_format="%.5f")
    ecm_tabs = {k: ecm_table(pulses, I) for k, I in PULSE_CHOICES.items()}
    ecm_tabs["I-dep"] = ecm_table_3d(pulses)
    for k, tab in ecm_tabs.items():
        tab.to_long().to_csv(PARAMS / f"ecm_2rc_{k}.csv", index=False, float_format="%.6g")

    # R_1s / R_10s summary (discharge pulses), mOhm
    dis = pulses[pulses.current > 0].copy()
    dis["c_rate"] = (dis.current / 3.0).round(1)
    dis["soc_pt"] = (dis.soc * 20).round() / 20
    r_tab = dis.pivot_table(index=["temp", "c_rate"], columns="soc_pt", values=["r_1s", "r_10s"]) * 1e3
    r_tab.to_csv(PARAMS / "r1s_r10s_mohm.csv", float_format="%.2f")

    # --- True SOC for drive cycles ----------------------------------------------------------
    inventory = []
    for T, cyc in itertools.product(TEMPS, DRIVE_CYCLES):
        rec = load(T, cyc)
        soc = 1.0 - rec.ah_dis / q_ref
        np.savez_compressed(
            PROC / f"{cyc}_{T}degC.npz", t=rec.t, i=rec.i, v=rec.v, temp_cell=rec.temp_cell, soc_true=soc, q_ref=q_ref
        )
        inventory.append(
            dict(temp=T, cycle=cyc, duration_s=rec.t[-1], dt_s=float(np.median(np.diff(rec.t))), soc_end=soc[-1],
                 i_rms=float(np.sqrt(np.mean(rec.i**2))), i_max=rec.i.max(), i_min=rec.i.min(),
                 tcell_min=rec.temp_cell.min(), tcell_max=rec.temp_cell.max(), v0=rec.v[0])
        )
    inv = pd.DataFrame(inventory)
    inv.to_csv(PROC / "drive_inventory.csv", index=False)

    # --- Model validation: 2RC + OCV with true SOC, voltage RMSE ----------------------------
    val = []
    examples = {}
    for T, cyc in itertools.product(TEMPS, VALIDATE_CYCLES):
        z = np.load(PROC / f"{cyc}_{T}degC.npz")
        for (ok, ot), (ek, et) in itertools.product(ocv_tabs.items(), ecm_tabs.items()):
            vm = simulate_tables(z["t"], z["i"], z["soc_true"], z["temp_cell"], ot, et)
            err = vm - z["v"]
            val.append(dict(temp=T, cycle=cyc, ocv=ok, pulse=ek, rmse_mv=1e3 * np.sqrt(np.mean(err**2)),
                            mean_mv=1e3 * err.mean(), maxabs_mv=1e3 * np.abs(err).max()))
            if cyc == "UDDS":
                examples[(T, ok, ek)] = (z["t"], z["v"], vm)
    val = pd.DataFrame(val)
    val.to_csv(REP / "01_model_validation.csv", index=False, float_format="%.3f")

    plots(ocv_avg, ocv_dis, rest_pts, pulses, ecm_tabs, examples, val)
    write_report(q_ref, pulses, inv, val, rest_pts, ocv_avg)
    print("phase 1 done")


def plots(ocv_avg, ocv_dis, rest_pts, pulses, ecm_tabs, examples, val):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for T in TEMPS:
        (l,) = ax[0].plot(SOC_GRID * 100, ocv_avg[T], label=f"{T} °C avg")
        ax[0].plot(SOC_GRID * 100, ocv_dis[T], "--", color=l.get_color(), lw=0.8)
        pts = np.array([(s, v) for TT, s, v in rest_pts if TT == T])
        ax[0].plot(pts[:, 0] * 100, pts[:, 1], "o", ms=3, color=l.get_color())
        ax[1].plot(SOC_GRID * 100, 1e3 * (ocv_avg[T] - ocv_dis[T]), color=l.get_color(), label=f"{T} °C")
    ax[0].set(xlabel="SOC [%]", ylabel="OCV [V]", title="OCV: C/20 avg (solid), C/20 dis (dashed), HPPC rest (dots)")
    ax[1].set(xlabel="SOC [%]", ylabel="avg - dis [mV]", title="Half-hysteresis estimate", ylim=(-10, 80))
    ax[0].legend(fontsize=7)
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "01_ocv.png", dpi=130)

    d = pulses[pulses.current > 0]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2), sharey=False)
    for T in TEMPS:
        dt = d[(d.temp == T) & (d.current.round() == 3)].sort_values("soc")
        for a, col in zip(ax, ("r0", "r_1s", "r_10s")):
            a.plot(dt.soc * 100, dt[col] * 1e3, "o-", ms=3, label=f"{T} °C")
    for a, title in zip(ax, ("R0 (0.1 s)", "R_1s", "R_10s (9.9 s)")):
        a.set(xlabel="SOC [%]", ylabel="mΩ", title=f"{title}, 1C discharge pulse", yscale="log")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "01_resistance.png", dpi=130)

    fig, ax = plt.subplots(2, 3, figsize=(15, 7))
    tab = ecm_tabs["1C"]
    for j, T in enumerate(tab.temps):
        for a, n in zip(ax.ravel(), ("r0", "r1", "tau1", "r2", "tau2")):
            scale = 1e3 if n.startswith("r") else 1
            a.plot(tab.soc * 100, tab.values[n][:, j] * scale, "o-", ms=3, label=f"{T:.0f} °C")
            a.set(title=n + (" [mΩ]" if scale == 1e3 else " [s]"), xlabel="SOC [%]")
    ax[0, 0].legend(fontsize=7)
    piv = val.groupby(["ocv", "pulse", "temp"]).rmse_mv.mean().unstack(["ocv", "pulse"])
    piv.plot(ax=ax[1, 2], marker="o")
    ax[1, 2].set(title="Drive-cycle voltage RMSE [mV]", xlabel="temp [°C]")
    fig.tight_layout()
    fig.savefig(FIG / "01_ecm_table_1C.png", dpi=130)

    best = val.groupby(["ocv", "pulse"]).rmse_mv.mean().idxmin()
    fig, ax = plt.subplots(len(TEMPS), 1, figsize=(12, 2.2 * len(TEMPS)), sharex=False)
    for a, T in zip(ax, TEMPS):
        t, v, vm = examples[(T, *best)]
        a.plot(t, v, lw=0.6, label="measured")
        a.plot(t, vm, lw=0.6, label=f"2RC (ocv={best[0]}, pulse={best[1]})")
        a.set(ylabel=f"{T} °C\nV")
    ax[0].legend(fontsize=7)
    ax[0].set_title("UDDS: measured vs 2RC model with true SOC")
    ax[-1].set_xlabel("t [s]")
    fig.tight_layout()
    fig.savefig(FIG / "01_udds_model_fit.png", dpi=110)
    plt.close("all")


def md_table(df: pd.DataFrame, fmt="{:.2f}") -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(fmt.format(x) if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(out)


def write_report(q_ref, pulses, inv, val, rest_pts, ocv_avg):
    d = pulses[pulses.current > 0]
    cnt = d.assign(c=(d.current / 3).round(1)).groupby(["temp", "c"]).size().unstack(fill_value=0)
    rest_err = []
    for T, s, v in rest_pts:
        rest_err.append(dict(temp=T, err_mv=1e3 * (v - np.interp(s, SOC_GRID, ocv_avg[T])), soc=s))
    rest_err = pd.DataFrame(rest_err)
    re = rest_err[rest_err.soc.between(0.1, 0.96)].groupby("temp").err_mv.agg(["mean", "min", "max"]).reset_index()
    summ = val.groupby(["ocv", "pulse"]).rmse_mv.agg(["mean", "max"]).reset_index()
    best = val.groupby(["ocv", "pulse"]).rmse_mv.mean().idxmin()
    per_t = val[val.pulse == "I-dep"].pivot_table(index="temp", columns=["ocv", "cycle"], values=["rmse_mv", "mean_mv"])
    per_t = per_t.round(1)
    per_t.columns = [f"{'RMSE' if a == 'rmse_mv' else 'bias'} {o} {c}" for a, o, c in per_t.columns]
    per_t = per_t[[c for c in per_t.columns if c.endswith(("UDDS", "US06"))]].reset_index()
    r = d[d.current.round() == 3].assign(soc_pt=lambda x: (x.soc * 20).round() / 20)
    r_mid = r[r.soc_pt.isin([0.2, 0.5, 0.8])].pivot_table(index="temp", columns="soc_pt", values=["r_1s", "r_10s"]) * 1e3
    r_mid.columns = [f"{a} @{b:.0%}" for a, b in r_mid.columns]
    r_mid = r_mid.reset_index()
    inv_s = inv.groupby("temp").agg(cycles=("cycle", "size"), soc_end_min=("soc_end", "min"), soc_end_max=("soc_end", "max"),
                                    tcell_min=("tcell_min", "min"), tcell_max=("tcell_max", "max")).reset_index()

    txt = f"""# 01. Dataset and model parameters (Phase 1)

Generated by `scripts/run_phase1.py`. Cell: Samsung INR21700-30T (NCM, 3 Ah), Kollmeyer / McMaster,
Mendeley Data 9xyvy2njj3 v2 (CC BY 4.0). Download with `scripts/download_30t.py`.

## Coverage against the requirements

| Requirement | Data | Status |
|---|---|---|
| 1.1 HPPC for R0, R1, τ1, R2, τ2 | 10 s discharge pulses at 1/2/6/12C (reduced at low T), ~13 SOC points, 6 temperatures | OK |
| 1.2 R_1s, R_10s | Same pulses, logged at 0.1 s. Pulse length is **9.9 s**, so R_10s is taken at 9.9 s | OK (9.9 s) |
| 1.3 Drive cycles with true SOC | UDDS, HWFET, LA92, US06, Mixed1–8 at 40/25/10/0/−10/−20 °C, 0.1 s logging, each from full charge | OK |

Discharge pulses found (count per temperature × C-rate):

{md_table(cnt.reset_index(), "{:.0f}")}

## SOC definition and true SOC

- Reference capacity Q_ref = C/20 discharge capacity at 25 °C = **{q_ref:.4f} Ah** (see docs/adr/0001).
- True SOC = 1 − (discharged Ah) / Q_ref, integrated from the full-charge start of each file.
  The tester logs the current of the interval that just ended; right-hold integration reproduces the
  tester's Ah counter exactly, trapezoidal integration does not (−0.2 Ah over HPPC).
- Drive cycles run until 95 % of the 1C capacity at that temperature, so the final SOC depends on T:

{md_table(inv_s)}

## OCV

- `params/ocv_soc_avg.csv`: mean of C/20 discharge and charge branches.
- `params/ocv_soc_dis.csv`: C/20 discharge branch + (C/20 current × R_10s), i.e. the discharge hysteresis branch.
- Check against HPPC 1 h rest voltages (10–96 % SOC), HPPC rest − OCV_avg [mV]:

{md_table(re)}

The rest points sit below the average curve at every temperature, consistent with discharge-side
hysteresis. At ~7 mV per 1 % SOC (mid SOC), a 15 mV OCV bias is ~2 %p SOC, so the OCV branch matters
for the 3 %p target. The model validation below picks the branch.

## Resistance (1C discharge pulse, mΩ)

{md_table(r_mid)}

Full table: `params/r1s_r10s_mohm.csv` (all C-rates, all SOC points). Plot: `reports/figures/01_resistance.png`.

## 2RC parameters

- Each discharge pulse is fitted from the rest before it up to the next discharge pulse (≤ 600 s after),
  with the OCV drift during the window included. Fits that hit a bound, or with RMSE > 3× the median at
  that temperature, are dropped and filled by SOC interpolation.
- Tables: `params/ecm_2rc_1C.csv`, `params/ecm_2rc_2C.csv` (SOC × T, one pulse current) and
  `params/ecm_2rc_I-dep.csv` (SOC × T × |I|, all pulse currents, log-|I| interpolation).
  All pulse fits: `params/hppc_pulses.csv`.
- At low temperature the polarisation resistance falls strongly with pulse current (−20 °C, 50 % SOC:
  R_10s ≈ 92 mΩ at 1C vs ≈ 68 mΩ at 4C), so the current-dependent table (`I-dep`) is needed.

## Model validation (true SOC in, voltage RMSE out)

The 2RC model with the measured cell temperature and the true SOC is run on UDDS, HWFET, LA92 and US06.
This measures model error alone, before any estimator.

Mean / max RMSE [mV] over all temperatures and cycles:

{md_table(summ)}

Lowest mean RMSE: OCV = `{best[0]}`, 2RC = `{best[1]}`. Current-dependent table, RMSE and bias
(model − measured) [mV] for UDDS and US06 with each OCV branch:

{md_table(per_t, "{:.1f}")}

### Findings

- The current-dependent table is clearly better than any single-current table at ≤ 0 °C.
- Error is still large at −10 / −20 °C (high-current US06 especially). Likely causes not modelled: the
  cell core heats faster than the case sensor during drive cycles, and resistance depends on current
  in a way 2RC with lookup only approximates. This is a known 2RC limit at low temperature.
- OCV branch: the discharge branch removes most of the voltage bias at 25 °C; the average branch has
  less bias at low temperature. Both are kept; **Phase 2 chooses by SOC error**, which is the target.
- What matters for SOC is mostly the low-frequency bias, not the dynamic RMSE: the EKF's
  measurement-noise R absorbs the dynamic error. Bias is what drives SOC error.

Plots: `reports/figures/01_ocv.png`, `01_ecm_table_1C.png`, `01_udds_model_fit.png`.
"""
    (REP / "01_dataset.md").write_text(txt)


if __name__ == "__main__":
    main()
