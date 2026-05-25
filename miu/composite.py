"""Constrained-OLS ETF composite that tracks a target index.

Solves: minimize ||X w - y||² subject to Σw = 1 and bounds[i] ≤ w[i] ≤ bounds[u].
Convex quadratic on a simplex, so SLSQP's local optimum is global.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from miu.config import MiuOptimizerError
from miu.etf import align_returns
from miu.proxy import TRADING_DAYS_PER_YEAR, ProxyMetric

log = logging.getLogger("miu.composite")


@dataclass
class CompositeResult:
    weights: dict[str, float]
    te: float  # annualized tracking error, decimal
    r2: float  # 1 - SSE / TSS (centered)
    residual_vol: float  # annualized stdev of residuals
    n_obs: int
    candidates_considered: list[ProxyMetric] = field(default_factory=list)
    solver_status: str = ""
    converged: bool = False


def top_k_by_te(metrics: list[ProxyMetric], k: int) -> list[str]:
    """Pick the k tickers with the lowest finite TE."""
    finite = [m for m in metrics if not math.isnan(m.te)]
    finite.sort(key=lambda m: m.te)
    return [m.ticker for m in finite[: max(1, k)]]


def _prune_constant_columns(panel: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose std is numerically zero (constant returns)."""
    stds = panel.std(ddof=0)
    keep = stds[stds > 1e-10].index
    return panel[keep]


def fit_composite(
    index_returns: pd.Series,
    etf_panel: pd.DataFrame,
    *,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    tol: float = 1e-12,
    max_iter: int = 500,
) -> CompositeResult:
    """Solve constrained OLS for the ETF weights that best track the index.

    `etf_panel` should already be filtered to the top-k candidates. The function
    inner-joins the panel against the index, prunes any all-constant columns,
    and returns the weight vector that minimizes squared residual error.
    """
    if etf_panel.empty or etf_panel.shape[1] == 0:
        raise MiuOptimizerError("composite: no candidate ETFs supplied")
    if not 0.0 <= min_weight <= max_weight <= 1.0:
        raise MiuOptimizerError(
            f"composite: invalid bounds min_weight={min_weight}, max_weight={max_weight}"
        )

    r_idx, panel = align_returns(index_returns, etf_panel)
    panel = _prune_constant_columns(panel)
    if panel.empty:
        raise MiuOptimizerError(
            "composite: no candidate ETFs have varying returns over the aligned window"
        )

    k = panel.shape[1]
    if max_weight * k < 1.0:
        raise MiuOptimizerError(
            f"composite: bounds infeasible — max_weight * k = {max_weight * k:.3f} < 1.0"
        )
    if min_weight * k > 1.0:
        raise MiuOptimizerError(
            f"composite: bounds infeasible — min_weight * k = {min_weight * k:.3f} > 1.0"
        )

    X = panel.values  # T × k
    y = r_idx.values  # T
    n_obs = X.shape[0]
    if n_obs < 2:
        raise MiuOptimizerError(f"composite: only {n_obs} aligned observations")

    def objective(w: np.ndarray) -> float:
        r = X @ w - y
        return float(r @ r)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * X.T @ (X @ w - y)

    x0 = np.full(k, 1.0 / k)
    bounds = [(min_weight, max_weight)] * k
    constraints = (
        {
            "type": "eq",
            "fun": lambda w: w.sum() - 1.0,
            "jac": lambda w: np.ones_like(w),
        },
    )
    res = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": tol, "maxiter": max_iter},
    )

    if not res.success:
        raise MiuOptimizerError(f"SLSQP failed: {res.message}")

    w = np.asarray(res.x, dtype=float)
    # Numerical hygiene: snap tiny weights to zero, then renormalize so Σw == 1.
    w[np.abs(w) < 1e-6] = 0.0
    total = w.sum()
    if total <= 0:
        raise MiuOptimizerError("SLSQP returned all-zero weights after snapping")
    w = w / total

    composite = X @ w
    diff = composite - y
    te = float(diff.std(ddof=0) * math.sqrt(TRADING_DAYS_PER_YEAR))
    residual_vol = te  # residual is (composite - index); same statistic
    tss = float(((y - y.mean()) ** 2).sum())
    sse = float((diff ** 2).sum())
    r2 = 1.0 - sse / tss if tss > 0 else float("nan")

    weights = {sym: float(wt) for sym, wt in zip(panel.columns, w, strict=True)}
    return CompositeResult(
        weights=weights,
        te=te,
        r2=r2,
        residual_vol=residual_vol,
        n_obs=n_obs,
        solver_status=str(res.message),
        converged=bool(res.success),
    )
