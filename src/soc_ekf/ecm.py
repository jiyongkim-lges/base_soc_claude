"""Second-order (2RC) Thevenin equivalent circuit model.

    v = OCV(SOC) - R0*i - v1 - v2,   dv_k/dt = -v_k/tau_k + i/C_k,  R_k = tau_k/C_k
with ``i`` discharge positive. Discretised exactly for a current held constant over each interval.
The 30T tester logs sample k with the current of the interval (t[k-1], t[k]], so the step into
sample k uses i[k].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass
class RC2:
    r0: float
    r1: float
    tau1: float
    r2: float
    tau2: float


def rc_voltages(t, i, r1, tau1, r2, tau2):
    """Polarisation voltages v1, v2, starting from rest (current i[k] held over (t[k-1], t[k]])."""
    dt = np.diff(t, prepend=t[0])
    a1, a2 = np.exp(-dt / tau1), np.exp(-dt / tau2)
    return _rc_recursion(a1, r1 * (1 - a1) * i), _rc_recursion(a2, r2 * (1 - a2) * i)


def _rc_recursion(a, u):
    """x[0] = 0, x[k] = a[k] x[k-1] + u[k]."""
    out = np.zeros(len(a))
    x = 0.0
    al, ul = np.broadcast_to(a, out.shape).tolist(), np.broadcast_to(u, out.shape).tolist()
    for k in range(1, len(out)):
        x = al[k] * x + ul[k]
        out[k] = x
    return out


def simulate(t, i, ocv, p: RC2):
    """Terminal voltage for constant parameters; ``ocv`` is an array aligned with ``t``."""
    v1, v2 = rc_voltages(t, i, p.r1, p.tau1, p.r2, p.tau2)
    return ocv - p.r0 * i - v1 - v2


@dataclass
class PulseFit:
    params: RC2
    rmse: float  # V


def fit_pulse(t, i, v, ocv, r0_guess: float) -> PulseFit:
    """Fit R0, R1, tau1, R2, tau2 to a pulse + relaxation window that starts at rest.

    ``ocv`` is the open-circuit voltage along the window (accounts for SOC drift during the pulse).
    Fitting is done in log-space for positivity; tau1 < tau2 is enforced by bounds
    (tau1 in [0.1, 10] s, tau2 in [10, 2000] s)."""

    def unpack(x):
        r0, r1, tau1, r2, tau2 = np.exp(x)
        return RC2(r0, r1, tau1, r2, tau2)

    def resid(x):
        return simulate(t, i, ocv, unpack(x)) - v

    r0g = max(r0_guess, 1e-4)
    x0 = np.log([r0g, 0.3 * r0g, 2.0, 0.5 * r0g, 60.0])
    lb = np.log([1e-4, 1e-5, 0.1, 1e-5, 10.0])
    ub = np.log([1.0, 1.0, 10.0, 1.0, 2000.0])
    sol = least_squares(resid, np.clip(x0, lb + 1e-9, ub - 1e-9), bounds=(lb, ub), x_scale="jac")
    p = unpack(sol.x)
    return PulseFit(p, float(np.sqrt(np.mean(sol.fun**2))))


def simulate_tables(t, i, soc, temp, ocv_tab, ecm_tab):
    """Terminal voltage with parameters looked up every step at (soc, temp), or at
    (soc, temp, |i|) when ``ecm_tab`` is an ``EcmTable3D``.

    Used to validate the model on drive cycles with the true SOC (no estimator involved)."""
    args = (soc, temp, i) if hasattr(ecm_tab, "i_grid") else (soc, temp)
    r0 = ecm_tab.at("r0", *args)
    r1, tau1 = ecm_tab.at("r1", *args), ecm_tab.at("tau1", *args)
    r2, tau2 = ecm_tab.at("r2", *args), ecm_tab.at("tau2", *args)
    ocv = ocv_tab.at("ocv", soc, temp)
    dt = np.diff(t, prepend=t[0])
    a1, a2 = np.exp(-dt / tau1), np.exp(-dt / tau2)
    v1 = _rc_recursion(a1, r1 * (1 - a1) * i)
    v2 = _rc_recursion(a2, r2 * (1 - a2) * i)
    return ocv - r0 * i - v1 - v2
