"""Universe registry schema and builder.

Public API
----------
    UniverseSpec     — Pydantic model for one entry in universes.yaml
    UniverseRegistry — container with .get() / .names()
    load_universes() — load + validate from YAML
    get_universe(name, date, con) -> pl.DataFrame

Output schema (same as build_universe_msf in session-1 notebook)
-----------------------------------------------------------------
    permno           Int64    CRSP permanent number
    permco           Int64
    ticker           Utf8
    cusip            Utf8
    name             Utf8
    shrcd            Int32
    exchcd           Int32
    siccd            Int32
    price_date       Date
    prc              Float64  abs(prc)
    shrout           Float64  shares outstanding (thousands)
    mcap_musd        Float64  market cap in $M = abs(prc) * shrout / 1000
    gvkey            Utf8     (null if no CCM match)
    funda_datadate   Date     (null if no Compustat fundamentals PIT-available)
    at               Float64  total assets
    lt               Float64  total liabilities
    seq              Float64  stockholders' equity
    ceq              Float64  common/ordinary equity
    sale             Float64  net sales / revenue
    revt             Float64  total revenue
    cogs             Float64  cost of goods sold
    ni               Float64  net income
    ebit             Float64  earnings before interest and tax
    ebitda           Float64  EBITDA
    csho             Float64  common shares outstanding (Compustat)
    mkvalt           Float64  market value (Compustat)
    oancf            Float64  operating cash flow
    capx             Float64  capital expenditure

Design notes
------------
price_source = "monthly"
    Uses crsp_msf for prices. Workaround for crsp_dsf only going to 2017 on
    Joshua's machine. Once crsp_dsf is refreshed, set price_source = "daily"
    in universes.yaml and the builder will switch automatically.

PIT correctness
    Fundamentals are gated by COALESCE(rdq, datadate + 90 days) <= asof,
    identical to the notebook workaround in session 1.

CCM link
    Joins via the ccm_link view built by optlab.links.build_link_views().
    One permno can map to multiple gvkeys across time; we keep the row with
    the most recent funda_datadate to avoid duplicates.

Ranking
    mcap_top_n: after applying base_filters, rank all surviving permnos by
        mcap_musd descending and keep the top_n. Permnos with null mcap are
        excluded (no price data).
    dolvol_top_n: compute average monthly dollar volume = avg(abs(prc) * vol)
        over the lookback_months period ending at asof, then rank and keep
        the top_n. Vol in crsp_msf is monthly share volume (units: hundreds
        of shares per CRSP documentation, but only relative ranking matters).
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import duckdb
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

import yaml

from optlab_research.logging_setup import get_logger

log = get_logger(__name__)

# ─── Schema ───────────────────────────────────────────────────────────────────


class BaseFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")

    shrcd: list[int] = Field(default_factory=lambda: [10, 11])
    exchcd: list[int] = Field(default_factory=lambda: [1, 2, 3])
    min_price: float = 1.0
    min_mcap_musd: float | None = None


class RankingSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["mcap_top_n", "dolvol_top_n"]
    top_n: int
    # Required only for dolvol_top_n
    lookback_months: int | None = None

    @model_validator(mode="after")
    def _check_dolvol_has_lookback(self) -> "RankingSpec":
        if self.method == "dolvol_top_n" and self.lookback_months is None:
            raise ValueError(
                "ranking.method='dolvol_top_n' requires ranking.lookback_months."
            )
        return self


class UniverseSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    price_source: Literal["monthly", "daily"] = "monthly"
    base_filters: BaseFilters = Field(default_factory=BaseFilters)
    ranking: RankingSpec | None = None


class UniverseRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    universes: list[UniverseSpec]

    @model_validator(mode="after")
    def _validate_unique_names(self) -> "UniverseRegistry":
        names = [u.name for u in self.universes]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"Duplicate universe names: {sorted(dupes)}")
        return self

    def get(self, name: str) -> UniverseSpec:
        for u in self.universes:
            if u.name == name:
                return u
        raise KeyError(
            f"Unknown universe {name!r}. "
            f"Registered universes: {self.names()}"
        )

    def names(self) -> list[str]:
        return [u.name for u in self.universes]

    def __len__(self) -> int:
        return len(self.universes)

    def __iter__(self):
        return iter(self.universes)


# ─── Loader ───────────────────────────────────────────────────────────────────

_DEFAULT_UNIVERSES_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "universes.yaml"
)

# Module-level cache — reloaded only if module is reloaded.
_registry: UniverseRegistry | None = None


def load_universes(path: Path | str | None = None) -> UniverseRegistry:
    """Load and validate the universe registry from universes.yaml."""
    p = Path(path) if path is not None else _DEFAULT_UNIVERSES_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"Universe registry not found at {p}. "
            "Ensure config/universes.yaml exists."
        )
    with p.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return UniverseRegistry.model_validate(raw)


def _get_registry() -> UniverseRegistry:
    global _registry
    if _registry is None:
        _registry = load_universes()
    return _registry


# ─── Core universe SQL ────────────────────────────────────────────────────────

_UNIVERSE_SQL_MONTHLY = """
WITH names AS (
    SELECT
        permno,
        permco,
        ticker,
        ncusip  AS cusip,
        comnam  AS name,
        shrcd,
        exchcd,
        siccd
    FROM crsp_stocknames
    WHERE DATE '{asof}' BETWEEN namedt::DATE
          AND COALESCE(nameenddt::DATE, DATE '9999-12-31')
      AND shrcd IN ({shrcd_list})
      AND exchcd IN ({exchcd_list})
),
not_delisted AS (
    SELECT n.* FROM names n
    WHERE NOT EXISTS (
        SELECT 1 FROM crsp_msedelist d
        WHERE d.permno = n.permno
          AND d.dlstdt::DATE <= DATE '{asof}'
    )
),
priced AS (
    SELECT
        n.*,
        p.date   AS price_date,
        ABS(p.prc) AS prc,
        p.shrout,
        ABS(p.prc) * p.shrout / 1000.0 AS mcap_musd
    FROM not_delisted n
    LEFT JOIN LATERAL (
        SELECT date, prc, shrout
        FROM crsp_msf
        WHERE permno = n.permno
          AND date::DATE <= DATE '{asof}'
        ORDER BY date DESC
        LIMIT 1
    ) p ON TRUE
),
linked AS (
    SELECT p.*, l.gvkey
    FROM priced p
    LEFT JOIN ccm_link l
        ON  l.permno = p.permno
        AND DATE '{asof}' BETWEEN l.start_dt AND l.end_dt
),
with_funda AS (
    SELECT
        u.*,
        f.datadate AS funda_datadate,
        f.at,  f.lt,    f.seq,   f.ceq,
        f.sale, f.revt, f.cogs,  f.ni,
        f.ebit, f.ebitda, f.csho, f.mkvalt,
        f.oancf, f.capx
    FROM linked u
    LEFT JOIN LATERAL (
        SELECT *
        FROM comp_funda
        WHERE gvkey = u.gvkey
          AND COALESCE(rdq::DATE, datadate::DATE + INTERVAL '90' DAY)
              <= DATE '{asof}'
        ORDER BY datadate DESC
        LIMIT 1
    ) f ON TRUE
)
SELECT * FROM with_funda
WHERE prc >= {min_price} OR prc IS NULL
"""

_UNIVERSE_SQL_DAILY = _UNIVERSE_SQL_MONTHLY.replace("crsp_msf", "crsp_dsf")


def _build_base_universe(
    con: duckdb.DuckDBPyConnection,
    asof: dt.date,
    spec: UniverseSpec,
) -> pl.DataFrame:
    """Run the base universe SQL and return a Polars DataFrame."""
    shrcd_list = ", ".join(str(x) for x in spec.base_filters.shrcd)
    exchcd_list = ", ".join(str(x) for x in spec.base_filters.exchcd)
    min_price = spec.base_filters.min_price

    sql_template = (
        _UNIVERSE_SQL_DAILY if spec.price_source == "daily" else _UNIVERSE_SQL_MONTHLY
    )
    sql = sql_template.format(
        asof=asof.isoformat(),
        shrcd_list=shrcd_list,
        exchcd_list=exchcd_list,
        min_price=min_price,
    )

    univ = con.execute(sql).pl()

    # CCM link can map one permno to multiple gvkeys across link windows.
    # Keep the row with the most recent funda_datadate to avoid duplicates.
    if univ["permno"].n_unique() < univ.height:
        univ = (
            univ
            .sort(
                ["permno", "funda_datadate", "gvkey"],
                descending=[False, True, False],
                nulls_last=True,
            )
            .unique(subset=["permno"], keep="first")
        )

    log.debug(
        "base universe as of %s: %d names (price_source=%s)",
        asof, univ.height, spec.price_source,
    )
    return univ


def _apply_min_mcap_filter(
    univ: pl.DataFrame, min_mcap_musd: float | None
) -> pl.DataFrame:
    if min_mcap_musd is None:
        return univ
    return univ.filter(
        pl.col("mcap_musd").is_not_null() & (pl.col("mcap_musd") >= min_mcap_musd)
    )


def _apply_mcap_top_n(univ: pl.DataFrame, top_n: int) -> pl.DataFrame:
    """Keep the top_n names by market cap. Permnos with null mcap are excluded."""
    return (
        univ
        .filter(pl.col("mcap_musd").is_not_null())
        .sort("mcap_musd", descending=True)
        .head(top_n)
    )


def _apply_dolvol_top_n(
    con: duckdb.DuckDBPyConnection,
    univ: pl.DataFrame,
    asof: dt.date,
    top_n: int,
    lookback_months: int,
) -> pl.DataFrame:
    """Rank by avg monthly dollar volume and keep top_n.

    Dollar volume per month = abs(prc) * vol.
    Vol in crsp_msf is monthly share volume in hundreds (per CRSP docs),
    so actual dollar volume = abs(prc) * vol * 100. The scaling cancels
    in the rank — we compute abs(prc)*vol for speed.
    """
    import calendar

    # Start date: first day of the month `lookback_months` ago.
    y = asof.year + (asof.month - lookback_months - 1) // 12
    m = (asof.month - lookback_months - 1) % 12 + 1
    start_date = dt.date(y, m, 1)

    permnos = univ["permno"].cast(pl.Int64).to_list()
    perm_df = pl.DataFrame({"permno": pl.Series(permnos, dtype=pl.Int64)})
    con.register("_dolvol_permnos_tmp", perm_df.to_arrow())

    try:
        sql = """
        SELECT
            m.permno,
            AVG(ABS(m.prc) * m.vol) AS avg_dolvol
        FROM  crsp_msf m
        INNER JOIN _dolvol_permnos_tmp p ON p.permno = m.permno
        WHERE m.date::DATE >= CAST(? AS DATE)
          AND m.date::DATE <= CAST(? AS DATE)
          AND m.prc IS NOT NULL
          AND m.vol IS NOT NULL
          AND m.vol > 0
        GROUP BY m.permno
        """
        dolvol = con.execute(sql, [start_date.isoformat(), asof.isoformat()]).pl()
    finally:
        con.unregister("_dolvol_permnos_tmp")

    top_permnos = (
        dolvol
        .sort("avg_dolvol", descending=True)
        .head(top_n)["permno"]
        .to_list()
    )

    result = univ.filter(pl.col("permno").is_in(top_permnos))
    log.debug(
        "dolvol_top_n: %d/%d permnos had vol history; keeping top %d",
        dolvol.height, len(permnos), top_n,
    )
    return result


# ─── Main entry point ─────────────────────────────────────────────────────────


def get_universe(
    name: str,
    date: str | dt.date,
    con: duckdb.DuckDBPyConnection,
) -> pl.DataFrame:
    """Build a named universe as of *date*.

    Parameters
    ----------
    name : str
        Universe name as defined in config/universes.yaml.
        e.g. "russell3000", "russell1000", "liquid_500", "tradeable".
    date : str or datetime.date
        As-of date (PIT). ISO string or date object.
    con  : duckdb.DuckDBPyConnection
        Open connection with optlab views registered.
        Use ``optlab_research.open_connection()`` to get one.

    Returns
    -------
    pl.DataFrame
        Universe DataFrame with the full column schema documented in this
        module's docstring. Schema is identical to ``build_universe_msf()``
        from the session-1 notebook.

    Raises
    ------
    KeyError   if *name* is not in universes.yaml.
    ValueError if the price_source view (crsp_msf / crsp_dsf) is not
               registered on *con*.

    Examples
    --------
    >>> import optlab_research as olr
    >>> from optlab_research.universes import get_universe
    >>> with olr.open_connection() as con:
    ...     univ = get_universe("liquid_500", "2023-12-29", con)
    ...     print(univ.shape)
    """
    if isinstance(date, str):
        asof = dt.date.fromisoformat(date)
    else:
        asof = date

    spec = _get_registry().get(name)
    log.info("building universe '%s' as of %s", name, asof.isoformat())

    # Verify the price-source view is available.
    price_view = "crsp_dsf" if spec.price_source == "daily" else "crsp_msf"
    views = con.execute(
        "SELECT view_name FROM duckdb_views() WHERE view_name = ?",
        [price_view],
    ).fetchall()
    if not views:
        raise ValueError(
            f"Universe '{name}' requires the '{price_view}' view, but it is not "
            f"registered on this connection. "
            f"{'Call optlab_research.open_connection() to auto-register views.' if price_view == 'crsp_msf' else 'Refresh crsp_dsf (optlab refresh --table crsp_dsf) and try again.'}"
        )

    # Step 1: base universe (SQL filter on shrcd, exchcd, min_price, PIT fundamentals)
    univ = _build_base_universe(con, asof, spec)

    # Step 2: optional min_mcap filter (applied before ranking)
    univ = _apply_min_mcap_filter(univ, spec.base_filters.min_mcap_musd)

    # Step 3: optional ranking to trim universe to top_n
    if spec.ranking is not None:
        r = spec.ranking
        if r.method == "mcap_top_n":
            univ = _apply_mcap_top_n(univ, r.top_n)
        elif r.method == "dolvol_top_n":
            univ = _apply_dolvol_top_n(
                con, univ, asof, r.top_n, r.lookback_months  # type: ignore[arg-type]
            )

    log.info(
        "universe '%s' as of %s: %d names | "
        "mcap non-null: %d | ceq non-null: %d",
        name, asof.isoformat(), univ.height,
        univ["mcap_musd"].is_not_null().sum(),
        univ["ceq"].is_not_null().sum() if "ceq" in univ.columns else 0,
    )

    return univ
