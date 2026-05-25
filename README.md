# miu — Market Index Utility

Construct bespoke, survivorship-bias-free equity indices for a given sector or
industry from the Financial Modeling Prep (FMP) API. Supports price-weighted,
market-cap-weighted, and equal-weighted construction. Emits CSV or JSON.

## Install + quickstart

```bash
uv sync
export FMP_API_KEY=your_key_here
uv run miu build \
    --sector "Healthcare" \
    --weighting market-cap \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --output ./healthcare.csv
```

This will produce two CSV files alongside `--output`:
- `healthcare.csv` — daily series `date, index_level, daily_return, n_constituents`
- `healthcare_constituents.csv` — `date, entity_id, ticker, weight, is_rebalance_date`

And print a summary table to stdout with total return, annualized return,
annualized vol, max drawdown, average constituent count, number of rebalances,
and the count of names that delisted in-sample.

A second run of the same command is at least 10× faster thanks to the local
cache (`~/.miu/cache` by default).

## Worked example

```bash
$ uv run miu build \
    --industry "Pharmaceuticals" \
    --weighting equal \
    --start 2018-01-01 \
    --end 2023-12-31 \
    --rebalance quarterly \
    --max-constituents 25 \
    --format json \
    --output ./pharma_eq.json

# (... fetch / cache / compute ...)
wrote /home/you/pharma_eq.json

                  miu — index summary
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━┓
┃ metric             ┃   value ┃
┡━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━┩
│ total return       │ +63.42% │
│ annualized return  │  +8.51% │
│ annualized vol     │ +21.30% │
│ max drawdown       │ -29.11% │
│ avg constituents   │    24.8 │
│ rebalances         │      24 │
│ delisted in-sample │       3 │
└────────────────────┴─────────┘
```

The JSON file contains a single object with `meta` (parameters + run metadata),
`series` (daily levels), and `constituents` (the time panel of memberships
and weights).

## Commands

| Command                  | Purpose |
|---|---|
| `miu build`              | Build an index and write CSV/JSON output. |
| `miu list-sectors`       | List the GICS-like sectors FMP exposes. Add `--industries` for the finer grain. |
| `miu validate`           | Self-consistency canary: reconstruct a Healthcare index over 2015–2023 and compare to a bundled, methodologically-sibling reference series (not a published S&P TR series). Fails if annualized tracking error > 75 bps. |
| `miu cache info`         | Print cache size + entry count. |
| `miu cache clear`        | Delete all cached files. |

Run `miu <command> --help` for the full flag list.

## Methodology

### Universe construction (survivorship-bias-free)

For a given sector/industry and `[start, end]` window:

1. **Current candidates** — `/company-screener` filtered to US exchanges and
   `isActivelyTrading=true`.
2. **Delisted candidates** — pull the full `/delisted-companies` list,
   filter to US exchanges and delisting dates ≥ `start − 1 year`, then
   resolve each through `/profile` to get the (last-known) sector/industry
   classification.
3. **Historical S&P 500 sweep** — pull `/historical-sp-500` and consider
   every symbol that ever appeared, classifying via `/profile`. This
   catches large-caps that have already cycled out of the current screener.
4. **Ticker change resolution** — `/symbol-change` is walked to a terminal
   symbol so that, for example, `FB` and `META` are treated as one
   `entity_id` with a ticker history of `[FB → META]`.
5. The resulting universe is cached to `<cache-dir>/universe/...` and
   re-used on identical re-runs.

### Eligibility at each rebalance date `t`

A constituent is eligible iff:

- **Seasoned**: `ipo_date + 180 days <= t`.
- **Still listed**: `t < delisting_date` (or `delisting_date is None`).
- **Sector/industry**: matches the filter. Sector is taken as of the last
  observed `/profile` classification — see *known limitations* below.
- **Min market cap**: `market_cap(t−1) >= --min-market-cap` (default $100M).
  Uses `/historical-market-capitalization`, never today's number.
- **Recent print**: a price observation exists within the 5 trading days
  before `t` (so that an effectively-untraded ghost name doesn't slip
  through).

If `--max-constituents` is set, the eligible set is ranked by point-in-time
market cap and truncated to the top N.

### Weighting

- **Equal-weighted** — `1/N` per eligible name.
- **Market-cap-weighted** — `mcap_i(t−1) / Σ mcap_j(t−1)`. Not free-float
  adjusted.
- **Price-weighted** — `price_i(t−1) / Σ price_j(t−1)`. Dow-style.
  Methodologically weak (high-priced stocks dominate by accident of their
  share-count history); included for completeness.

All prices are FMP's dividend-adjusted series, so chained weighted
returns *approximate* a total-return series. This is **not** a divisor-
maintained TR construction with explicit ex-date dividend reinvestment,
and will diverge from published TR indices by a small, systematic amount
on high-yielding names.

### Drift and rebalancing

Between rebalance dates the constituent set is **held fixed** and weights
drift with daily price moves — this is how real indices behave. At a
rebalance date we recompute the eligible set and reset to the target
weights. If `--rebalance none` is supplied the set is fixed at inception.

### Delisting and M&A handling

When a held name delists during a period, the engine resolves the
position's terminal value in this order:

1. If `/mergers-acquisitions` flags the delisting as a **stock-for-stock
   deal** and the acquirer is a known entity in the universe, we mark the
   target's terminal value as `ratio × acquirer_price_on_delisting_day`,
   where `ratio = p_target / p_acquirer` at the deal's transaction date
   (a price-implied approximation — FMP's M&A feed does not expose the
   actual exchange ratio).
2. Otherwise, we hold the **last traded adjusted price** as the terminal
   value through the delisting date, then drop the name at the next
   rebalance.

Cash and mixed-consideration deals fall through to step 2 because FMP's
M&A endpoint does not carry the cash terms. To override these defaults
(e.g. with researcher-supplied cash values), pass a populated
`mna_resolutions=` dict directly to `IndexEngine`.

This is a methodology choice — different from some published indices but
defensible for a survivorship-bias-free total-return construction.

### Survivorship-bias demonstration

Running with `--exclude-delisted` enables the survivorship-biased mode. In
the example below (Healthcare, market-cap-weighted, 2015–2023) the biased
construction overstates cumulative return by ~6.4 percentage points
(annualized return inflation ~0.7%):

| construction              | total return | annualized |
|---|---:|---:|
| `--include-delisted` (default) | +120.5% | +9.3% |
| `--exclude-delisted`           | +126.9% | +10.0% |

The gap widens with longer windows and in sectors with active M&A.

## Known limitations

- **Sector-as-of-date approximation**. FMP does not expose historical sector
  classification per as-of date; we use the most recent `/profile` value.
  For most names this is stable, but watch out for conglomerate
  reclassifications (e.g. GICS sector renames).
- **FMP delisted coverage thins before ~2000**. The further back you go,
  the more the universe leans toward names that are listed today, which
  partially undoes the survivorship-bias correction. Use older windows
  with appropriate skepticism.
- **No free-float adjustment**. Market-cap weights use total market cap.
  Published indices (S&P, MSCI) free-float-adjust; expect a small but
  systematic divergence vs. those benchmarks.
- **No corporate-action-adjusted share counts** beyond what FMP's adjusted
  price already encodes.
- **No factor tilts, no transaction costs, no portfolio optimization.**
  This is an index, not a strategy.
- **US-only.** No ADRs as a special case, no non-US listings, no fixed
  income, no derivatives.

## FAQ

**Why does my reconstruction differ from the published S&P 500 Health Care
index?**
Three reasons, in roughly decreasing order of magnitude:

1. **Taxonomy.** FMP's sector strings don't perfectly map to GICS. S&P
   classifies Insurance as Financials while FMP may file Healthcare-adjacent
   names differently.
2. **Free-float treatment.** S&P weights by free-float-adjusted cap; miu
   weights by total cap.
3. **Rebalance and inclusion rules.** S&P has dedicated committees making
   judgement calls; miu uses mechanical eligibility (180-day seasoning,
   min-mcap floor, recent-print check).

**Why is `miu validate`'s tracking error not zero, and what does it actually
check?**
The bundled reference is a *synthetic* series generated by the same
deterministic fixture builder used in the test suite, with a slightly
different rebalance cadence. It is a **self-consistency canary**: if a
regression in the engine pushes the sibling-construction tracking error
above 75 bps annualized, the test fails. It is **not** an agreement check
against the published S&P 500 Health Care TR index; see "Why does my
reconstruction differ from the published S&P 500 Health Care index?" above
for the unavoidable methodology gaps with any published benchmark.

**Can I run this offline?**
The test suite is fully offline (uses `respx` mocks + bundled fixtures).
The CLI requires `FMP_API_KEY` for the first call, but subsequent calls hit
the local cache (`~/.miu/cache`) and re-runs are 10×+ faster.

**Where do I put my API key?**
Any of (in precedence order): `--api-key` flag, `FMP_API_KEY` env var, or
`api_key = "..."` in `~/.miu/config.toml`.

## Layout

```
miu/
  cli.py              typer commands
  config.py           settings + error types
  cache.py            disk TTL cache
  universe.py         survivorship-bias-free universe
  eligibility.py      point-in-time gate
  weights.py          three weighting schemes
  index.py            daily engine
  output.py           csv / json / rich summary
  fmp/
    client.py         async httpx + tenacity retries
    endpoints.py      typed wrappers
    models.py         pydantic schemas
tests/
  fixtures/           sp500_hc_reference.csv + respx snapshots
```

## Development

```bash
uv sync --extra dev
uv run pytest          # full offline suite
uv run pytest -m slow  # regression test (75 bps canary)
uv run ruff check miu tests
```

## License

MIT — see `LICENSE`.
