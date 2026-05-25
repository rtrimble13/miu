"""CSV / JSON writers and the rich summary table (spec §8)."""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from miu import __version__
from miu.composite import CompositeResult
from miu.index import IndexResult
from miu.proxy import ProxyMetric


def write_csv(result: IndexResult, output: Path) -> tuple[Path, Path]:
    """Writes <output>.csv (series) and <output>_constituents.csv (panel)."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        series_path = output
        const_path = output.with_name(f"{output.stem}_constituents.csv")
    else:
        series_path = output.with_suffix(".csv")
        const_path = output.with_suffix("").with_name(f"{output.stem}_constituents.csv")
    result.series.to_csv(series_path, index=False)
    result.constituents.to_csv(const_path, index=False)
    return series_path, const_path


def write_json(result: IndexResult, output: Path, meta: dict[str, Any]) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() != ".json":
        output = output.with_suffix(".json")
    body = {
        "meta": {
            **meta,
            "constituent_count_avg": float(result.summary.get("avg_constituents", 0.0)),
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "miu_version": __version__,
        },
        "series": [
            {
                "date": _iso(row["date"]),
                "level": float(row["index_level"]),
                "return": float(row["daily_return"]),
                "n": int(row["n_constituents"]),
            }
            for row in result.series.to_dict(orient="records")
        ],
        "constituents": [
            {
                "date": _iso(row["date"]),
                "entity_id": row["entity_id"],
                "ticker": row["ticker"],
                "weight": float(row["weight"]),
                "is_rebalance_date": bool(row["is_rebalance_date"]),
            }
            for row in result.constituents.to_dict(orient="records")
        ],
    }
    output.write_text(json.dumps(body, indent=2, default=str))
    return output


def print_summary(result: IndexResult, console: Console | None = None) -> None:
    console = console or Console()
    s = result.summary
    table = Table(title="miu — index summary", show_header=True, header_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_row("total return", _pct(s["total_return"]))
    table.add_row("annualized return", _pct(s["annualized_return"]))
    table.add_row("annualized vol", _pct(s["annualized_vol"]))
    table.add_row("max drawdown", _pct(s["max_drawdown"]))
    table.add_row("avg constituents", f"{s['avg_constituents']:.1f}")
    table.add_row("rebalances", f"{int(s['rebalances'])}")
    table.add_row("delisted in-sample", f"{int(s['delisted_in_sample'])}")
    console.print(table)


def _iso(d: Any) -> str:
    if isinstance(d, str):
        return d
    if hasattr(d, "isoformat"):
        return d.isoformat()
    return str(d)


def _pct(x: float) -> str:
    return f"{100.0 * x:+.2f}%"


def series_dataframe(result: IndexResult) -> pd.DataFrame:
    """Convenience accessor for callers who want the raw frame."""
    return result.series


# ---------- recommend / composite output ----------


def _meta_path(output: Path) -> Path:
    base = output.with_suffix("")
    return base.parent / f"{base.name}_meta.json"


def _metric_row(m: ProxyMetric, is_winner: bool) -> dict[str, Any]:
    return {
        "ticker": m.ticker,
        "name": m.name,
        "te": _none_or_float(m.te),
        "corr": _none_or_float(m.corr),
        "beta": _none_or_float(m.beta),
        "r2": _none_or_float(m.r2),
        "expense_ratio": _none_or_float(m.expense_ratio),
        "aum": _none_or_float(m.aum),
        "n_obs": int(m.n_obs),
        "is_winner": bool(is_winner),
    }


def _none_or_float(x: float | None) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return float(x)


def write_proxy_table(
    metrics: list[ProxyMetric],
    winner: ProxyMetric | None,
    output: Path,
    meta: dict[str, Any],
    *,
    fmt: str = "csv",
) -> tuple[Path, Path | None]:
    """Write the recommend table.

    CSV mode: writes `output` (table) and `<output>_meta.json` (sidecar).
    JSON mode: writes a single file containing both meta and table; sidecar = None.
    """
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    winner_ticker = winner.ticker if winner else None
    rows = [_metric_row(m, is_winner=(m.ticker == winner_ticker)) for m in metrics]
    enriched_meta = {
        **meta,
        "winner": winner_ticker,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "miu_version": __version__,
    }
    if fmt == "json":
        path = output if output.suffix.lower() == ".json" else output.with_suffix(".json")
        body = {"meta": enriched_meta, "winner": winner_ticker, "table": rows}
        path.write_text(json.dumps(body, indent=2, default=str))
        return path, None
    # csv
    path = output if output.suffix.lower() == ".csv" else output.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    meta_path = _meta_path(path)
    meta_path.write_text(json.dumps(enriched_meta, indent=2, default=str))
    return path, meta_path


def write_composite(
    result: CompositeResult,
    output: Path,
    meta: dict[str, Any],
    *,
    fmt: str = "csv",
) -> tuple[Path, Path | None]:
    """Write the composite weights table + a fit-summary sidecar."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    weight_rows = [
        {"ticker": sym, "weight": float(w)}
        for sym, w in sorted(result.weights.items(), key=lambda kv: -kv[1])
    ]
    fit_meta = {
        **meta,
        "fit": {
            "te": _none_or_float(result.te),
            "te_bps": _none_or_float(result.te * 1e4),
            "r2": _none_or_float(result.r2),
            "residual_vol": _none_or_float(result.residual_vol),
            "residual_vol_bps": _none_or_float(result.residual_vol * 1e4),
            "n_obs": int(result.n_obs),
            "solver_status": result.solver_status,
            "converged": bool(result.converged),
        },
        "candidates_considered": [
            _metric_row(m, is_winner=False) for m in result.candidates_considered
        ],
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "miu_version": __version__,
    }
    if fmt == "json":
        path = output if output.suffix.lower() == ".json" else output.with_suffix(".json")
        body = {"meta": fit_meta, "weights": weight_rows}
        path.write_text(json.dumps(body, indent=2, default=str))
        return path, None
    path = output if output.suffix.lower() == ".csv" else output.with_suffix(".csv")
    pd.DataFrame(weight_rows).to_csv(path, index=False)
    meta_path = _meta_path(path)
    meta_path.write_text(json.dumps(fit_meta, indent=2, default=str))
    return path, meta_path


def print_proxy_table(
    metrics: list[ProxyMetric],
    winner: ProxyMetric | None,
    console: Console | None = None,
) -> None:
    console = console or Console()
    table = Table(title="miu — ETF proxy candidates", show_header=True, header_style="bold")
    table.add_column("ticker")
    table.add_column("name")
    table.add_column("TE (bps)", justify="right")
    table.add_column("corr", justify="right")
    table.add_column("beta", justify="right")
    table.add_column("R²", justify="right")
    table.add_column("ER", justify="right")
    table.add_column("AUM ($M)", justify="right")
    table.add_column("n", justify="right")
    winner_ticker = winner.ticker if winner else None
    for m in metrics:
        style = "bold green" if m.ticker == winner_ticker else None
        table.add_row(
            m.ticker,
            (m.name or "")[:32],
            _bps(m.te),
            _num(m.corr),
            _num(m.beta),
            _num(m.r2),
            _expense(m.expense_ratio),
            _aum_m(m.aum),
            str(m.n_obs),
            style=style,
        )
    console.print(table)


def print_composite(result: CompositeResult, console: Console | None = None) -> None:
    console = console or Console()
    weights_table = Table(title="miu — composite weights", show_header=True, header_style="bold")
    weights_table.add_column("ticker")
    weights_table.add_column("weight", justify="right")
    for sym, w in sorted(result.weights.items(), key=lambda kv: -kv[1]):
        weights_table.add_row(sym, f"{w:.4f}")
    console.print(weights_table)

    fit_table = Table(title="miu — composite fit", show_header=True, header_style="bold")
    fit_table.add_column("metric")
    fit_table.add_column("value", justify="right")
    fit_table.add_row("TE (bps annualized)", _bps(result.te))
    fit_table.add_row("residual vol (bps annualized)", _bps(result.residual_vol))
    fit_table.add_row("R²", _num(result.r2))
    fit_table.add_row("observations", str(result.n_obs))
    fit_table.add_row("solver status", result.solver_status or "—")
    fit_table.add_row("converged", "yes" if result.converged else "no")
    console.print(fit_table)


def _bps(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 1e4:,.1f}"


def _num(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:.4f}"


def _aum_m(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x / 1e6:,.0f}"


def _expense(x: float | None) -> str:
    """Format an expense ratio (decimal, e.g. 0.0009) as `0.09%`."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    # FMP sometimes reports expense ratio as percent (e.g. 0.09) and sometimes
    # as decimal (e.g. 0.0009). Heuristic: anything >= 0.1 is treated as already
    # a percent; otherwise multiply by 100.
    pct = x if x >= 0.1 else x * 100.0
    return f"{pct:.2f}%"
