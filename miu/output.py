"""CSV / JSON writers and the rich summary table (spec §8)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from rich.console import Console
from rich.table import Table

from miu import __version__
from miu.index import IndexResult


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
