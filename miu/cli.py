"""Typer CLI surface (spec §2)."""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path

import typer
from rich.console import Console

from miu import __version__
from miu.cache import DiskCache
from miu.config import (
    DEFAULT_CACHE_DIR,
    MiuApiError,
    MiuConfigError,
    MiuUniverseError,
    Settings,
)
from miu.fmp import endpoints as ep
from miu.fmp.client import FmpClient
from miu.index import EngineConfig, IndexEngine
from miu.output import print_summary, write_csv, write_json
from miu.universe import Constituent, UniverseRequest, build_universe, known_sectors

app = typer.Typer(
    name="miu",
    help="Market Index Utility — build bespoke equity indices from FMP data.",
    no_args_is_help=True,
    add_completion=False,
)
cache_app = typer.Typer(help="Local cache hygiene.")
app.add_typer(cache_app, name="cache")

err = Console(stderr=True)
out = Console()


def _setup_logging(verbose: bool, fmt: str) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    if fmt == "json":
        logging.basicConfig(
            level=level,
            stream=sys.stderr,
            format='{"ts":"%(asctime)s","level":"%(levelname)s","name":"%(name)s","msg":%(message)r}',
        )
    else:
        logging.basicConfig(
            level=level,
            stream=sys.stderr,
            format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
        )


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"miu {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool | None = typer.Option(
        None,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print version and exit.",
    ),
) -> None:
    pass


@app.command()
def build(
    sector: str | None = typer.Option(None, "--sector", help="GICS-like sector name."),
    industry: str | None = typer.Option(None, "--industry", help="Finer-grained industry name."),
    weighting: str = typer.Option(
        ..., "--weighting", help="One of: price | market-cap | equal."
    ),
    start: str = typer.Option(..., "--start", help="Start date YYYY-MM-DD."),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD (default: today)."),
    rebalance: str = typer.Option("quarterly", "--rebalance"),
    base_value: float = typer.Option(1000.0, "--base-value"),
    format: str = typer.Option("csv", "--format", help="csv | json"),
    output: Path = typer.Option(..., "--output", help="Output file path."),
    exchange: str = typer.Option("all", "--exchange", help="NYSE | NASDAQ | AMEX | all"),
    min_market_cap: int = typer.Option(100_000_000, "--min-market-cap"),
    max_constituents: int | None = typer.Option(None, "--max-constituents"),
    include_delisted: bool = typer.Option(
        True,
        "--include-delisted/--exclude-delisted",
        help="Default includes delisted names. --exclude-delisted is the survivorship-biased mode.",
    ),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    api_key: str | None = typer.Option(None, "--api-key"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    log_format: str = typer.Option("human", "--log-format", help="human | json"),
) -> None:
    """Build an index and write CSV or JSON output."""
    _setup_logging(verbose, log_format)
    if (sector and industry) or (not sector and not industry):
        raise typer.BadParameter("Specify exactly one of --sector or --industry.")
    if weighting not in {"price", "market-cap", "equal"}:
        raise typer.BadParameter("--weighting must be one of: price, market-cap, equal")
    if rebalance not in {"monthly", "quarterly", "annual", "none"}:
        raise typer.BadParameter("--rebalance must be one of: monthly, quarterly, annual, none")
    if format not in {"csv", "json"}:
        raise typer.BadParameter("--format must be csv or json")

    start_d = _parse_date(start)
    end_d = _parse_date(end) if end else date.today()
    if end_d <= start_d:
        raise typer.BadParameter("--end must be after --start")

    if not include_delisted:
        err.print(
            "[yellow]warning:[/yellow] --exclude-delisted enables the survivorship-biased mode. "
            "Returns will be systematically overstated."
        )

    exchanges = _resolve_exchanges(exchange)
    settings = Settings.load(
        api_key=api_key, cache_dir=cache_dir, verbose=verbose, log_format=log_format
    )

    try:
        asyncio.run(
            _build_async(
                settings=settings,
                sector=sector,
                industry=industry,
                weighting=weighting,
                start=start_d,
                end=end_d,
                rebalance=rebalance,
                base_value=base_value,
                output_format=format,
                output=output,
                exchanges=tuple(exchanges),
                min_market_cap=float(min_market_cap),
                max_constituents=max_constituents,
                include_delisted=include_delisted,
            )
        )
    except MiuConfigError as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(2) from exc
    except (MiuApiError, MiuUniverseError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc


async def _build_async(
    *,
    settings: Settings,
    sector: str | None,
    industry: str | None,
    weighting: str,
    start: date,
    end: date,
    rebalance: str,
    base_value: float,
    output_format: str,
    output: Path,
    exchanges: tuple[str, ...],
    min_market_cap: float,
    max_constituents: int | None,
    include_delisted: bool,
) -> None:
    async with FmpClient(settings) as client:
        await _validate_sector(client, sector, industry)
        request = UniverseRequest(
            sector=sector,
            industry=industry,
            start=start,
            end=end,
            exchanges=exchanges,
            include_delisted=include_delisted,
        )
        constituents = await build_universe(request, client)
        if not constituents:
            raise MiuUniverseError(
                f"no constituents found for sector={sector!r} industry={industry!r}"
            )

        prices, market_caps = await _load_panels(client, constituents, start, end)

        config = EngineConfig(
            weighting=weighting,
            start=start,
            end=end,
            rebalance=rebalance,  # type: ignore[arg-type]
            base_value=base_value,
            min_market_cap=min_market_cap,
            max_constituents=max_constituents,
            sector=sector,
            industry=industry,
        )
        engine = IndexEngine(constituents, prices, market_caps, config)
        result = engine.run()

    if output_format == "csv":
        sp, cp = write_csv(result, output)
        out.print(f"wrote {sp} and {cp}")
    else:
        meta = {
            "sector": sector,
            "industry": industry,
            "weighting": weighting,
            "start": str(start),
            "end": str(end),
            "rebalance": rebalance,
            "base_value": base_value,
        }
        path = write_json(result, output, meta)
        out.print(f"wrote {path}")
    print_summary(result, out)


async def _load_panels(
    client: FmpClient,
    constituents: list[Constituent],
    start: date,
    end: date,
) -> tuple[dict[str, dict[date, float]], dict[str, dict[date, float]]]:
    prices: dict[str, dict[date, float]] = {}
    market_caps: dict[str, dict[date, float]] = {}
    import asyncio as _aio

    async def load_one(c: Constituent) -> None:
        all_prices: dict[date, float] = {}
        all_mcaps: dict[date, float] = {}
        for span in c.ticker_history:
            sym = span.ticker
            rows = await ep.historical_prices(client, sym, start=start, end=end)
            for r in rows:
                if r.price is not None:
                    all_prices.setdefault(r.date, r.price)
            mrows = await ep.historical_market_cap(client, sym, start=start, end=end)
            for m in mrows:
                all_mcaps.setdefault(m.date, m.market_cap)
        prices[c.entity_id] = all_prices
        market_caps[c.entity_id] = all_mcaps

    await _aio.gather(*(load_one(c) for c in constituents))
    return prices, market_caps


async def _validate_sector(client: FmpClient, sector: str | None, industry: str | None) -> None:
    """If sector/industry isn't in FMP's taxonomy, raise with did-you-mean."""
    if not sector and not industry:
        return
    try:
        rows = await ep.screener(client, sector=sector, industry=industry, limit=1)
    except MiuApiError:
        return
    if rows:
        return
    candidates = known_sectors() if sector else []
    if sector and candidates:
        match = difflib.get_close_matches(sector, candidates, n=1, cutoff=0.6)
        hint = f" did you mean {match[0]!r}?" if match else ""
        raise MiuUniverseError(f"unknown sector {sector!r}.{hint}")


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as exc:
        raise typer.BadParameter(f"invalid date {s!r}, expected YYYY-MM-DD") from exc


def _resolve_exchanges(value: str) -> list[str]:
    v = value.strip().lower()
    if v == "all":
        return ["NYSE", "NASDAQ", "AMEX"]
    norm = value.strip().upper()
    if norm not in {"NYSE", "NASDAQ", "AMEX"}:
        raise typer.BadParameter("--exchange must be NYSE, NASDAQ, AMEX, or all")
    return [norm]


@app.command("list-sectors")
def list_sectors(
    industries: bool = typer.Option(False, "--industries", help="List industries instead of sectors."),
    api_key: str | None = typer.Option(None, "--api-key"),
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
) -> None:
    """Print available sectors (or --industries) from FMP."""
    settings = Settings.load(api_key=api_key, cache_dir=cache_dir)
    asyncio.run(_list_sectors_async(settings, industries))


async def _list_sectors_async(settings: Settings, industries: bool) -> None:
    async with FmpClient(settings) as client:
        rows = await ep.screener(client, limit=10000)
        names = sorted({(r.industry if industries else r.sector) or "" for r in rows})
        for n in names:
            if n:
                out.print(n)


@app.command()
def validate(
    cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir"),
    api_key: str | None = typer.Option(None, "--api-key"),
    reference: Path = typer.Option(
        Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "sp500_hc_reference.csv",
        "--reference",
        help="Reference S&P 500 Health Care daily levels CSV.",
    ),
) -> None:
    """Reconstruct S&P 500 Health Care and compare to a bundled reference."""
    settings = Settings.load(api_key=api_key, cache_dir=cache_dir)
    try:
        te = asyncio.run(_validate_async(settings, reference))
    except MiuConfigError as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(2) from exc
    threshold_bps = 75.0
    out.print(f"tracking error: {te * 10000:.1f} bps (annualized)")
    if te * 10000 > threshold_bps:
        err.print(f"[red]FAIL[/red] tracking error exceeds {threshold_bps} bps")
        raise typer.Exit(1)
    out.print(f"[green]OK[/green] tracking error within {threshold_bps} bps")


async def _validate_async(settings: Settings, reference: Path) -> float:
    import math

    import pandas as pd

    async with FmpClient(settings) as client:
        req = UniverseRequest(
            sector="Healthcare",
            industry=None,
            start=date(2015, 1, 1),
            end=date(2023, 12, 31),
            exchanges=("NYSE", "NASDAQ", "AMEX"),
            include_delisted=True,
        )
        constituents = await build_universe(req, client)
        prices, market_caps = await _load_panels(client, constituents, req.start, req.end)
        engine = IndexEngine(
            constituents,
            prices,
            market_caps,
            EngineConfig(
                weighting="market-cap",
                start=req.start,
                end=req.end,
                rebalance="quarterly",
                base_value=1000.0,
                sector="Healthcare",
            ),
        )
        result = engine.run()

    ref = pd.read_csv(reference, parse_dates=["date"])
    ref["date"] = ref["date"].dt.date
    series = result.series.copy()
    merged = series.merge(ref, on="date", suffixes=("_miu", "_ref"))
    if merged.empty:
        raise MiuUniverseError("no overlap between miu series and reference fixture")
    merged["r_miu"] = merged["index_level"].pct_change()
    merged["r_ref"] = merged["level"].pct_change()
    diff = (merged["r_miu"] - merged["r_ref"]).dropna()
    if diff.empty:
        return 0.0
    return float(diff.std(ddof=0) * math.sqrt(252))


@cache_app.command("info")
def cache_info(cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir")) -> None:
    stats = DiskCache(cache_dir).stats()
    out.print(json.dumps({"cache_dir": str(cache_dir), **stats}, indent=2))


@cache_app.command("clear")
def cache_clear(cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir")) -> None:
    removed = DiskCache(cache_dir).clear()
    out.print(f"removed {removed} cached files from {cache_dir}")
