"""Pydantic v2 models for FMP API responses.

We model only the fields we use, keep `extra="ignore"` to absorb FMP's
churn, and normalize date-like fields to `datetime.date`.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _parse_date(value: Any) -> date | None:
    if value in (None, "", "0000-00-00"):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    s = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    # Last-ditch: ISO 8601
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).date()
    except ValueError as exc:
        raise ValueError(f"unparseable date: {value!r}") from exc


class _Base(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True, str_strip_whitespace=True)


class ScreenerRow(_Base):
    symbol: str
    company_name: str | None = Field(default=None, alias="companyName")
    market_cap: float | None = Field(default=None, alias="marketCap")
    sector: str | None = None
    industry: str | None = None
    exchange: str | None = None
    exchange_short_name: str | None = Field(default=None, alias="exchangeShortName")
    country: str | None = None
    is_actively_trading: bool | None = Field(default=None, alias="isActivelyTrading")


class DelistedRow(_Base):
    symbol: str
    company_name: str | None = Field(default=None, alias="companyName")
    exchange: str | None = None
    ipo_date: date | None = Field(default=None, alias="ipoDate")
    delisted_date: date | None = Field(default=None, alias="delistedDate")

    @field_validator("ipo_date", "delisted_date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> Any:
        return _parse_date(v)


class SymbolChange(_Base):
    date: date
    name: str | None = None
    old_symbol: str = Field(alias="oldSymbol")
    new_symbol: str = Field(alias="newSymbol")

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> Any:
        return _parse_date(v)


class Profile(_Base):
    symbol: str
    company_name: str | None = Field(default=None, alias="companyName")
    cik: str | None = None
    sector: str | None = None
    industry: str | None = None
    exchange: str | None = None
    exchange_short_name: str | None = Field(default=None, alias="exchangeShortName")
    country: str | None = None
    currency: str | None = None
    ipo_date: date | None = Field(default=None, alias="ipoDate")
    is_actively_trading: bool | None = Field(default=None, alias="isActivelyTrading")
    is_etf: bool | None = Field(default=None, alias="isEtf")
    is_fund: bool | None = Field(default=None, alias="isFund")
    is_adr: bool | None = Field(default=None, alias="isAdr")

    @field_validator("ipo_date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> Any:
        return _parse_date(v)


class HistoricalPrice(_Base):
    date: date
    adj_close: float | None = Field(default=None, alias="adjClose")
    close: float | None = None
    volume: float | None = None

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> Any:
        return _parse_date(v)

    @property
    def price(self) -> float | None:
        return self.adj_close if self.adj_close is not None else self.close


class HistoricalMarketCap(_Base):
    symbol: str | None = None
    date: date
    market_cap: float = Field(alias="marketCap")

    @field_validator("date", mode="before")
    @classmethod
    def _date(cls, v: Any) -> Any:
        return _parse_date(v)


class MnaEvent(_Base):
    symbol: str | None = None
    targeted_company_name: str | None = Field(default=None, alias="targetedCompanyName")
    targeted_symbol: str | None = Field(default=None, alias="targetedSymbol")
    transaction_date: date | None = Field(default=None, alias="transactionDate")
    acceptance_time: date | None = Field(default=None, alias="acceptanceTime")
    deal_type: str | None = Field(default=None, alias="dealType")
    consideration: str | None = None  # e.g. "cash" / "stock" / "mixed"

    @field_validator("transaction_date", "acceptance_time", mode="before")
    @classmethod
    def _date(cls, v: Any) -> Any:
        return _parse_date(v)


class SP500Membership(_Base):
    """A row from /historical-sp-500. FMP returns one event per add/remove."""

    date_added: date | None = Field(default=None, alias="dateAdded")
    added_security: str | None = Field(default=None, alias="addedSecurity")
    removed_ticker: str | None = Field(default=None, alias="removedTicker")
    removed_security: str | None = Field(default=None, alias="removedSecurity")
    symbol: str | None = None
    sector: str | None = None
    sub_sector: str | None = Field(default=None, alias="subSector")
    reason: str | None = None

    @field_validator("date_added", mode="before")
    @classmethod
    def _date(cls, v: Any) -> Any:
        return _parse_date(v)
