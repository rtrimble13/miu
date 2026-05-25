# miu — Market Index Utility

Construct bespoke, survivorship-bias-free equity indices for a given sector or
industry from the Financial Modeling Prep (FMP) API. Supports
price-weighted, market-cap-weighted, and equal-weighted construction. Emits
CSV or JSON.

> Status: **scaffolding only** — full implementation lands in the next push.

## Quickstart

```bash
uv sync
export FMP_API_KEY=...
uv run miu build \
  --sector "Health Care" \
  --weighting market-cap \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --output ./health_care.csv
```

## Layout

```
miu/
  cli.py              # typer app, command surface
  config.py           # env, API key, defaults
  cache.py            # local file cache for FMP calls
  universe.py         # build the survivorship-bias-free universe
  eligibility.py      # point-in-time eligibility checks
  weights.py          # three weighting schemes
  index.py            # the index calculation engine
  output.py           # csv/json writers
  fmp/
    client.py         # async httpx client with retries + rate limiting
    endpoints.py      # typed endpoint wrappers
    models.py         # pydantic schemas for FMP responses
tests/
```

## Methodology

Filled in alongside the implementation. Will cover: how the universe is built,
eligibility rules at each rebalance, delisting / M&A handling, the
sector-as-of-date approximation, FMP coverage limits before ~2000, no
free-float adjustment, and how this index will differ from published
S&P / MSCI sector indices.

## License

MIT — see `LICENSE`.
