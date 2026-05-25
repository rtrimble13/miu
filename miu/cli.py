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
from miu.composite import fit_composite, top_k_by_te
from miu.config import (
    DEFAULT_CACHE_DIR,
    MiuApiError,
    MiuConfigError,
    MiuOptimizerError,
    MiuUniverseError,
    Settings,
)
from miu.etf import (
    IndexReturns,
    discover_etf_candidates,
    load_etf_profiles,
    load_etf_returns_panel,
    load_index_from_file,
)
from miu.fmp import endpoints as ep
from miu.fmp.client import FmpClient
from miu.fmp.models import MnaEvent
from miu.index import EngineConfig, IndexEngine, IndexResult, MnaResolution
from miu.output import (
    print_composite,
    print_proxy_table,
    print_summary,
    write_composite,
    write_csv,
    write_json,
    write_proxy_table,
)
from miu.proxy import rank_proxies
from miu.proxy import recommend as _rank_recommend
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


async def _build_index_result(
    client: FmpClient,
    *,
    sector: str | None,
    industry: str | None,
    weighting: str,
    start: date,
    end: date,
    rebalance: str,
    base_value: float,
    exchanges: tuple[str, ...],
    min_market_cap: float,
    max_constituents: int | None,
    include_delisted: bool,
) -> IndexResult:
    """Build the IndexResult given an open client. Shared by `build` and the
    inline-mode `recommend` / `composite` commands.
    """
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
    mna_events = await ep.mergers_acquisitions(client, start=start, end=end)
    mna_resolutions = _resolve_mna(mna_events, constituents, prices)
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
    engine = IndexEngine(
        constituents, prices, market_caps, config, mna_resolutions=mna_resolutions
    )
    return engine.run()


def _index_result_to_returns(
    result: IndexResult, meta: dict[str, object]
) -> IndexReturns:
    """Project an IndexResult into the IndexReturns shape consumed by proxy/composite."""
    import pandas as pd

    s = result.series
    if s.empty:
        raise MiuUniverseError("index produced no rows")
    rets = pd.Series(
        s["daily_return"].astype(float).values,
        index=pd.Index([d for d in s["date"]], name="date"),
        name="index_return",
    )
    rets = rets.iloc[1:]  # first row is by construction 0; drop for honest stats
    return IndexReturns(returns=rets, meta=dict(meta))


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
        result = await _build_index_result(
            client,
            sector=sector,
            industry=industry,
            weighting=weighting,
            start=start,
            end=end,
            rebalance=rebalance,
            base_value=base_value,
            exchanges=exchanges,
            min_market_cap=min_market_cap,
            max_constituents=max_constituents,
            include_delisted=include_delisted,
        )

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


def _resolve_mna(
    events: list[MnaEvent],
    constituents: list[Constituent],
    prices: dict[str, dict[date, float]],
) -> dict[str, MnaResolution]:
    """Build entity_id -> MnaResolution from FMP's /mergers-acquisitions feed.

    FMP's M&A response carries deal metadata (target, acquirer, dates,
    deal_type / consideration string) but not the cash terms or share-exchange
    ratio. We can therefore resolve stock-for-stock deals using a price-ratio
    proxy at the deal's transaction date (1 target share ≈ (p_target/p_acq)
    acquirer shares), provided the acquirer is itself a known entity in the
    universe. Cash and mixed deals are left without a resolution; the engine
    falls back to last-traded price in that case.
    """
    ticker_to_entity: dict[str, str] = {}
    for c in constituents:
        for span in c.ticker_history:
            ticker_to_entity[span.ticker.upper()] = c.entity_id

    out: dict[str, MnaResolution] = {}
    for ev in events:
        target_sym = (ev.targeted_symbol or "").upper()
        if not target_sym:
            continue
        target_id = ticker_to_entity.get(target_sym)
        if not target_id or target_id in out:
            continue
        consid = ((ev.consideration or "") + " " + (ev.deal_type or "")).lower()
        is_stock = "stock" in consid or "share" in consid or "exchange" in consid
        if not is_stock:
            continue
        acquirer_sym = (ev.symbol or "").upper()
        acquirer_id = ticker_to_entity.get(acquirer_sym)
        if not acquirer_id or acquirer_id == target_id:
            continue
        deal_date = ev.transaction_date or ev.acceptance_time
        if deal_date is None:
            continue
        p_target = _price_at(prices.get(target_id, {}), deal_date)
        p_acquirer = _price_at(prices.get(acquirer_id, {}), deal_date)
        if not p_target or not p_acquirer or p_acquirer <= 0:
            continue
        ratio = p_target / p_acquirer
        out[target_id] = MnaResolution(acquirer_id=acquirer_id, ratio=ratio)
    return out


def _price_at(series: dict[date, float], when: date) -> float | None:
    if not series:
        return None
    best: tuple[date, float] | None = None
    for d, v in series.items():
        if d <= when and (best is None or d > best[0]):
            best = (d, v)
    return None if best is None else best[1]


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
    """Self-consistency canary against a bundled (synthetic) reference series."""
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
        mna_events = await ep.mergers_acquisitions(client, start=req.start, end=req.end)
        mna_resolutions = _resolve_mna(mna_events, constituents, prices)
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
            mna_resolutions=mna_resolutions,
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


def _parse_tickers(value: str | None) -> list[str]:
    if not value:
        return []
    out: list[str] = []
    for tok in value.replace(";", ",").split(","):
        tok = tok.strip().upper()
        if tok:
            out.append(tok)
    return out


def _validate_index_input_flags(
    *,
    index_path: Path | None,
    sector: str | None,
    industry: str | None,
    weighting: str | None,
    rebalance: str | None,
    exchange: str | None,
    min_market_cap: int | None,
    include_delisted_set: bool,
) -> None:
    """Enforce the mutual exclusivity between --index and inline-build flags."""
    inline_flags = {
        "--sector": sector is not None,
        "--industry": industry is not None,
        "--weighting": weighting is not None,
        "--rebalance": rebalance is not None,
        "--exchange": exchange is not None,
        "--min-market-cap": min_market_cap is not None,
        "--include-delisted/--exclude-delisted": include_delisted_set,
    }
    set_inline = [k for k, v in inline_flags.items() if v]
    if index_path is not None and set_inline:
        raise typer.BadParameter(
            f"--index is mutually exclusive with inline-build flags: {', '.join(set_inline)}"
        )
    if index_path is None:
        if not (sector or industry):
            raise typer.BadParameter("Specify --index, or one of --sector / --industry.")
        if sector and industry:
            raise typer.BadParameter("Specify exactly one of --sector or --industry.")
        if weighting is None:
            raise typer.BadParameter("--weighting is required when --index is not used.")


async def _resolve_target(
    client: FmpClient,
    *,
    index_path: Path | None,
    sector: str | None,
    industry: str | None,
    weighting: str | None,
    start: date | None,
    end: date | None,
    rebalance: str,
    exchanges: tuple[str, ...],
    min_market_cap: float,
    include_delisted: bool,
) -> IndexReturns:
    """Materialize the IndexReturns from either an --index file or an inline build.

    Caller owns the client's lifecycle (use `async with FmpClient(...) as client:`).
    """
    if index_path is not None:
        ir = load_index_from_file(index_path)
        ir = ir.slice(start, end)
        if start is None or end is None:
            start = start or ir.returns.index.min()
            end = end or ir.returns.index.max()
        ir.meta.setdefault("source", str(index_path))
        ir.meta.setdefault("start", str(start))
        ir.meta.setdefault("end", str(end))
        return ir

    assert weighting is not None and start is not None and end is not None
    result = await _build_index_result(
        client,
        sector=sector,
        industry=industry,
        weighting=weighting,
        start=start,
        end=end,
        rebalance=rebalance,
        base_value=1000.0,
        exchanges=exchanges,
        min_market_cap=min_market_cap,
        max_constituents=None,
        include_delisted=include_delisted,
    )
    meta: dict[str, object] = {
        "sector": sector,
        "industry": industry,
        "weighting": weighting,
        "rebalance": rebalance,
        "start": str(start),
        "end": str(end),
    }
    return _index_result_to_returns(result, meta)


@app.command()
def recommend(
    index: Path | None = typer.Option(
        None,
        "--index",
        help="Path to a previously-built index CSV/JSON (from `miu build`). "
        "Mutually exclusive with --sector/--industry/--weighting.",
    ),
    sector: str | None = typer.Option(None, "--sector", help="Inline-build sector."),
    industry: str | None = typer.Option(None, "--industry", help="Inline-build industry."),
    weighting: str | None = typer.Option(
        None, "--weighting", help="One of: price | market-cap | equal."
    ),
    start: str | None = typer.Option(
        None, "--start", help="Start date YYYY-MM-DD (required for inline; optional with --index)."
    ),
    end: str | None = typer.Option(None, "--end", help="End date YYYY-MM-DD."),
    rebalance: str | None = typer.Option(None, "--rebalance"),
    exchange: str | None = typer.Option(None, "--exchange", help="NYSE | NASDAQ | AMEX | all."),
    min_market_cap: int | None = typer.Option(None, "--min-market-cap"),
    include_delisted: bool = typer.Option(
        True,
        "--include-delisted/--exclude-delisted",
        help="Default includes delisted names. --exclude-delisted is survivorship-biased.",
    ),
    candidates: str | None = typer.Option(
        None,
        "--candidates",
        help="Comma-separated ETF tickers; overrides auto-discovery entirely.",
    ),
    add_candidates: str | None = typer.Option(
        None, "--add-candidates", help="Extra ETF tickers to add to auto-discovery."
    ),
    max_candidates: int = typer.Option(25, "--max-candidates"),
    min_etf_aum: float = typer.Option(1e8, "--min-etf-aum"),
    min_overlap: int = typer.Option(60, "--min-overlap"),
    format: str = typer.Option("csv", "--format", help="csv | json"),
    output: Path = typer.Option(..., "--output", help="Output file path."),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    api_key: str | None = typer.Option(None, "--api-key"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    log_format: str = typer.Option("human", "--log-format"),
) -> None:
    """Recommend a single ETF proxy for the target index (lowest tracking error)."""
    _setup_logging(verbose, log_format)
    include_delisted_set = (
        "--exclude-delisted" in sys.argv or "--include-delisted" in sys.argv
    )
    _validate_index_input_flags(
        index_path=index,
        sector=sector,
        industry=industry,
        weighting=weighting,
        rebalance=rebalance,
        exchange=exchange,
        min_market_cap=min_market_cap,
        include_delisted_set=include_delisted_set,
    )
    if weighting is not None and weighting not in {"price", "market-cap", "equal"}:
        raise typer.BadParameter("--weighting must be one of: price, market-cap, equal")
    rebalance_resolved = rebalance or "quarterly"
    if rebalance_resolved not in {"monthly", "quarterly", "annual", "none"}:
        raise typer.BadParameter("--rebalance must be one of: monthly, quarterly, annual, none")
    if format not in {"csv", "json"}:
        raise typer.BadParameter("--format must be csv or json")
    if index is None and not start:
        raise typer.BadParameter("--start is required when --index is not used")
    start_d = _parse_date(start) if start else None
    end_d = _parse_date(end) if end else (date.today() if start_d and index is None else None)
    if start_d and end_d and end_d <= start_d:
        raise typer.BadParameter("--end must be after --start")
    exchanges = tuple(_resolve_exchanges(exchange or "all"))
    settings = Settings.load(
        api_key=api_key, cache_dir=cache_dir, verbose=verbose, log_format=log_format
    )

    try:
        asyncio.run(
            _recommend_async(
                settings=settings,
                index_path=index,
                sector=sector,
                industry=industry,
                weighting=weighting,
                start=start_d,
                end=end_d,
                rebalance=rebalance_resolved,
                exchanges=exchanges,
                min_market_cap=(
                    float(min_market_cap) if min_market_cap is not None else 100_000_000.0
                ),
                include_delisted=include_delisted,
                candidates_override=_parse_tickers(candidates),
                add_candidates=_parse_tickers(add_candidates),
                max_candidates=max_candidates,
                min_etf_aum=min_etf_aum,
                min_overlap=min_overlap,
                output_format=format,
                output=output,
            )
        )
    except MiuConfigError as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(2) from exc
    except (MiuApiError, MiuUniverseError, MiuOptimizerError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc


async def _recommend_async(
    *,
    settings: Settings,
    index_path: Path | None,
    sector: str | None,
    industry: str | None,
    weighting: str | None,
    start: date | None,
    end: date | None,
    rebalance: str,
    exchanges: tuple[str, ...],
    min_market_cap: float,
    include_delisted: bool,
    candidates_override: list[str],
    add_candidates: list[str],
    max_candidates: int,
    min_etf_aum: float,
    min_overlap: int,
    output_format: str,
    output: Path,
) -> None:
    async with FmpClient(settings) as client:
        ir = await _resolve_target(
            client,
            index_path=index_path,
            sector=sector,
            industry=industry,
            weighting=weighting,
            start=start,
            end=end,
            rebalance=rebalance,
            exchanges=exchanges,
            min_market_cap=min_market_cap,
            include_delisted=include_delisted,
        )
        candidate_tickers, source = await _resolve_candidates(
            client,
            ir,
            sector=sector or _meta_sector(ir),
            industry=industry or _meta_industry(ir),
            exchanges=exchanges,
            candidates_override=candidates_override,
            add_candidates=add_candidates,
            max_candidates=max_candidates,
            min_etf_aum=min_etf_aum,
        )
        if not candidate_tickers:
            raise MiuUniverseError("no ETF candidates found; pass --candidates explicitly")

        start_window = start or ir.returns.index.min()
        end_window = end or ir.returns.index.max()
        panel = await load_etf_returns_panel(
            client, candidate_tickers, start=start_window, end=end_window
        )
        profiles = await load_etf_profiles(client, candidate_tickers)

    if panel.empty:
        raise MiuUniverseError("no ETF price data returned for any candidate")

    winner, table = _rank_recommend(ir.returns, panel, profiles, min_overlap=min_overlap)

    meta = {
        "target_index": ir.meta,
        "candidate_source": source,
        "candidates_requested": candidate_tickers,
        "min_overlap": min_overlap,
        "trading_days_used": int(len(ir.returns)),
    }
    if output_format == "json":
        path, _ = write_proxy_table(table, winner, output, meta, fmt="json")
        out.print(f"wrote {path}")
    else:
        path, meta_path = write_proxy_table(table, winner, output, meta, fmt="csv")
        out.print(f"wrote {path}" + (f" and {meta_path}" if meta_path else ""))

    print_proxy_table(table, winner, out)
    if winner is None:
        err.print(
            "[yellow]warning:[/yellow] no candidate met the minimum overlap requirement."
        )


@app.command()
def composite(
    index: Path | None = typer.Option(None, "--index"),
    sector: str | None = typer.Option(None, "--sector"),
    industry: str | None = typer.Option(None, "--industry"),
    weighting: str | None = typer.Option(None, "--weighting"),
    start: str | None = typer.Option(None, "--start"),
    end: str | None = typer.Option(None, "--end"),
    rebalance: str | None = typer.Option(None, "--rebalance"),
    exchange: str | None = typer.Option(None, "--exchange"),
    min_market_cap: int | None = typer.Option(None, "--min-market-cap"),
    include_delisted: bool = typer.Option(
        True, "--include-delisted/--exclude-delisted"
    ),
    candidates: str | None = typer.Option(None, "--candidates"),
    add_candidates: str | None = typer.Option(None, "--add-candidates"),
    max_candidates: int = typer.Option(25, "--max-candidates"),
    min_etf_aum: float = typer.Option(1e8, "--min-etf-aum"),
    min_overlap: int = typer.Option(60, "--min-overlap"),
    top_k: int = typer.Option(5, "--top-k", help="Top-k candidates by individual TE."),
    min_weight: float = typer.Option(0.0, "--min-weight"),
    max_weight: float = typer.Option(1.0, "--max-weight"),
    format: str = typer.Option("csv", "--format"),
    output: Path = typer.Option(..., "--output"),
    cache_dir: Path | None = typer.Option(None, "--cache-dir"),
    api_key: str | None = typer.Option(None, "--api-key"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
    log_format: str = typer.Option("human", "--log-format"),
) -> None:
    """Fit a constrained-OLS composite of ETFs that tracks the target index."""
    _setup_logging(verbose, log_format)
    include_delisted_set = (
        "--exclude-delisted" in sys.argv or "--include-delisted" in sys.argv
    )
    _validate_index_input_flags(
        index_path=index,
        sector=sector,
        industry=industry,
        weighting=weighting,
        rebalance=rebalance,
        exchange=exchange,
        min_market_cap=min_market_cap,
        include_delisted_set=include_delisted_set,
    )
    if weighting is not None and weighting not in {"price", "market-cap", "equal"}:
        raise typer.BadParameter("--weighting must be one of: price, market-cap, equal")
    rebalance_resolved = rebalance or "quarterly"
    if rebalance_resolved not in {"monthly", "quarterly", "annual", "none"}:
        raise typer.BadParameter("--rebalance must be one of: monthly, quarterly, annual, none")
    if format not in {"csv", "json"}:
        raise typer.BadParameter("--format must be csv or json")
    if not 0.0 <= min_weight <= max_weight <= 1.0:
        raise typer.BadParameter("require 0 <= --min-weight <= --max-weight <= 1")
    if top_k < 1:
        raise typer.BadParameter("--top-k must be >= 1")
    if index is None and not start:
        raise typer.BadParameter("--start is required when --index is not used")
    start_d = _parse_date(start) if start else None
    end_d = _parse_date(end) if end else (date.today() if start_d and index is None else None)
    if start_d and end_d and end_d <= start_d:
        raise typer.BadParameter("--end must be after --start")
    exchanges = tuple(_resolve_exchanges(exchange or "all"))
    settings = Settings.load(
        api_key=api_key, cache_dir=cache_dir, verbose=verbose, log_format=log_format
    )

    try:
        asyncio.run(
            _composite_async(
                settings=settings,
                index_path=index,
                sector=sector,
                industry=industry,
                weighting=weighting,
                start=start_d,
                end=end_d,
                rebalance=rebalance_resolved,
                exchanges=exchanges,
                min_market_cap=(
                    float(min_market_cap) if min_market_cap is not None else 100_000_000.0
                ),
                include_delisted=include_delisted,
                candidates_override=_parse_tickers(candidates),
                add_candidates=_parse_tickers(add_candidates),
                max_candidates=max_candidates,
                min_etf_aum=min_etf_aum,
                min_overlap=min_overlap,
                top_k=top_k,
                min_weight=min_weight,
                max_weight=max_weight,
                output_format=format,
                output=output,
            )
        )
    except MiuConfigError as exc:
        err.print(f"[red]config error:[/red] {exc}")
        raise typer.Exit(2) from exc
    except (MiuApiError, MiuUniverseError, MiuOptimizerError) as exc:
        err.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(1) from exc


async def _composite_async(
    *,
    settings: Settings,
    index_path: Path | None,
    sector: str | None,
    industry: str | None,
    weighting: str | None,
    start: date | None,
    end: date | None,
    rebalance: str,
    exchanges: tuple[str, ...],
    min_market_cap: float,
    include_delisted: bool,
    candidates_override: list[str],
    add_candidates: list[str],
    max_candidates: int,
    min_etf_aum: float,
    min_overlap: int,
    top_k: int,
    min_weight: float,
    max_weight: float,
    output_format: str,
    output: Path,
) -> None:
    async with FmpClient(settings) as client:
        ir = await _resolve_target(
            client,
            index_path=index_path,
            sector=sector,
            industry=industry,
            weighting=weighting,
            start=start,
            end=end,
            rebalance=rebalance,
            exchanges=exchanges,
            min_market_cap=min_market_cap,
            include_delisted=include_delisted,
        )
        candidate_tickers, source = await _resolve_candidates(
            client,
            ir,
            sector=sector or _meta_sector(ir),
            industry=industry or _meta_industry(ir),
            exchanges=exchanges,
            candidates_override=candidates_override,
            add_candidates=add_candidates,
            max_candidates=max_candidates,
            min_etf_aum=min_etf_aum,
        )
        if not candidate_tickers:
            raise MiuUniverseError("no ETF candidates found; pass --candidates explicitly")

        start_window = start or ir.returns.index.min()
        end_window = end or ir.returns.index.max()
        panel = await load_etf_returns_panel(
            client, candidate_tickers, start=start_window, end=end_window
        )
        profiles = await load_etf_profiles(client, candidate_tickers)

    if panel.empty:
        raise MiuUniverseError("no ETF price data returned for any candidate")

    full_metrics = rank_proxies(ir.returns, panel, profiles, min_overlap=min_overlap)
    selected = top_k_by_te(full_metrics, top_k)
    if not selected:
        raise MiuUniverseError("no ETF candidate met the minimum overlap requirement")
    sub_panel = panel[selected]
    result = fit_composite(
        ir.returns,
        sub_panel,
        min_weight=min_weight,
        max_weight=max_weight,
    )
    result.candidates_considered = [m for m in full_metrics if m.ticker in selected]

    meta = {
        "target_index": ir.meta,
        "candidate_source": source,
        "candidates_requested": candidate_tickers,
        "top_k": top_k,
        "min_overlap": min_overlap,
        "min_weight": min_weight,
        "max_weight": max_weight,
        "trading_days_used": int(len(ir.returns)),
    }
    if output_format == "json":
        path, _ = write_composite(result, output, meta, fmt="json")
        out.print(f"wrote {path}")
    else:
        path, meta_path = write_composite(result, output, meta, fmt="csv")
        out.print(f"wrote {path}" + (f" and {meta_path}" if meta_path else ""))

    print_composite(result, out)


async def _resolve_candidates(
    client: FmpClient,
    ir: IndexReturns,
    *,
    sector: str | None,
    industry: str | None,
    exchanges: tuple[str, ...],
    candidates_override: list[str],
    add_candidates: list[str],
    max_candidates: int,
    min_etf_aum: float,
) -> tuple[list[str], str]:
    """Resolve the final candidate ticker list and describe its provenance."""
    if candidates_override:
        extras = [t for t in add_candidates if t not in candidates_override]
        return candidates_override + extras, "user-supplied"

    if not (sector or industry):
        # No way to auto-discover without a sector/industry hint; require user list.
        if add_candidates:
            return add_candidates, "user-supplied (no sector hint)"
        raise typer.BadParameter(
            "--candidates is required when the target index has no sector/industry hint"
        )

    discovered = await discover_etf_candidates(
        client,
        sector=sector,
        industry=industry,
        exchanges=list(exchanges) if exchanges else None,
        min_aum=min_etf_aum,
        max_candidates=max_candidates,
    )
    if add_candidates:
        seen = set(discovered)
        for t in add_candidates:
            if t not in seen:
                discovered.append(t)
                seen.add(t)
    return discovered, f"auto (sector={sector!r}, industry={industry!r})"


def _meta_sector(ir: IndexReturns) -> str | None:
    v = ir.meta.get("sector")
    return str(v) if v else None


def _meta_industry(ir: IndexReturns) -> str | None:
    v = ir.meta.get("industry")
    return str(v) if v else None


@cache_app.command("info")
def cache_info(cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir")) -> None:
    stats = DiskCache(cache_dir).stats()
    out.print(json.dumps({"cache_dir": str(cache_dir), **stats}, indent=2))


@cache_app.command("clear")
def cache_clear(cache_dir: Path = typer.Option(DEFAULT_CACHE_DIR, "--cache-dir")) -> None:
    removed = DiskCache(cache_dir).clear()
    out.print(f"removed {removed} cached files from {cache_dir}")
