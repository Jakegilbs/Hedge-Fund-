"""Insider purchases from the SEC's free Insider Transactions Data Sets (Forms 3, 4 and 5).

The SEC publishes every insider filing as structured data, one zip file per
quarter since 2006:
https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets

This module downloads each quarter once, keeps only open-market PURCHASES
(transaction code "P": an insider paying cash for shares at the market price,
not grants, option exercises or sales), and caches them in
data/cache/insiders/. The filing date is when the purchase became public, so a
backtest that acts after the filing date never uses information early.

The SEC asks every automated client to identify itself. Put a line like this in
the project's .env file:
    SEC_USER_AGENT=Your Name your.email@example.com
"""
from __future__ import annotations

import io
import os
import time
import zipfile
from datetime import date

import pandas as pd

from .config import DATA_DIR

QUARTER_URL = "https://www.sec.gov/files/structureddata/data/form-345-data-sets/{q}_form345.zip"
CACHE = DATA_DIR / "cache" / "insiders"
COLUMNS = ["accession", "filing_date", "trans_date", "ticker", "issuer", "issuer_cik", "owner", "owner_cik",
           "relationship", "title", "shares", "price", "value", "owned_after", "direct"]


def user_agent() -> str:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if not ua or "@" not in ua:
        raise SystemExit("The SEC requires a contact in every request. Add a line to the .env file in the "
                         "HedgeFund folder:\n    SEC_USER_AGENT=Your Name your.email@example.com")
    return ua


def quarters(start_year: int, today: date | None = None) -> list[str]:
    today = today or date.today()
    out = []
    for y in range(start_year, today.year + 1):
        for q in range(1, 5):
            if (y, q) < (today.year, (today.month - 1) // 3 + 1):        # only finished quarters
                out.append(f"{y}q{q}")
    return out


def _table(z: zipfile.ZipFile, name: str) -> pd.DataFrame:
    """Read one TSV from the zip, whatever folder or letter case it uses."""
    match = next((n for n in z.namelist() if n.lower().endswith(name.lower())), None)
    if match is None:
        raise ValueError(f"{name} not found in the zip (files: {', '.join(z.namelist()[:8])})")
    with z.open(match) as f:
        df = pd.read_csv(f, sep="\t", dtype=str, keep_default_na=False, na_values=[""],
                         quoting=3, on_bad_lines="skip", encoding_errors="replace")
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def _date(s: pd.Series) -> pd.Series:
    """SEC files write dates as 02-JAN-2024; accept ISO dates too."""
    d = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    return d.fillna(pd.to_datetime(s, errors="coerce", format="mixed"))


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.str.replace(",", "", regex=False), errors="coerce")


def parse_quarter(data: bytes) -> pd.DataFrame:
    """Open-market purchases reported on original Form 4 filings in one quarterly zip."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        sub = _table(z, "SUBMISSION.tsv")
        own = _table(z, "REPORTINGOWNER.tsv")
        trans = _table(z, "NONDERIV_TRANS.tsv")
    t = trans[(trans["TRANS_CODE"].str.strip().str.upper() == "P")
              & (trans["TRANS_ACQUIRED_DISP_CD"].str.strip().str.upper() == "A")].copy()
    t["shares"] = _num(t["TRANS_SHARES"])
    t["price"] = _num(t["TRANS_PRICEPERSHARE"])
    t = t[(t.shares > 0) & (t.price > 0)]
    sub = sub[sub["DOCUMENT_TYPE"].str.strip() == "4"]
    own = own.drop_duplicates("ACCESSION_NUMBER")          # joint filings: count the purchase once
    df = t.merge(sub, on="ACCESSION_NUMBER").merge(own, on="ACCESSION_NUMBER", how="left")
    out = pd.DataFrame({
        "accession": df["ACCESSION_NUMBER"],
        "filing_date": _date(df["FILING_DATE"]),
        "trans_date": _date(df["TRANS_DATE"]),
        "ticker": df["ISSUERTRADINGSYMBOL"].fillna("").str.strip().str.upper().str.replace(".", "-", regex=False),
        "issuer": df["ISSUERNAME"], "issuer_cik": df["ISSUERCIK"],
        "owner": df.get("RPTOWNERNAME"), "owner_cik": df.get("RPTOWNERCIK"),
        "relationship": df.get("RPTOWNER_RELATIONSHIP"), "title": df.get("RPTOWNER_TITLE"),
        "shares": df["shares"], "price": df["price"], "value": df["shares"] * df["price"],
        "owned_after": _num(df["SHRS_OWND_FOLWNG_TRANS"]) if "SHRS_OWND_FOLWNG_TRANS" in df else None,
        "direct": df.get("DIRECT_INDIRECT_OWNERSHIP"),
    })
    bad = {"", "NONE", "N/A", "NA", "NULL"}
    return out[out.filing_date.notna() & ~out.ticker.isin(bad)][COLUMNS].reset_index(drop=True)


def _download(q: str, ua: str) -> bytes | None:
    import urllib.error
    import urllib.request
    req = urllib.request.Request(QUARTER_URL.format(q=q), headers={"User-Agent": ua})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None                                  # not published yet
        raise


def load_purchases(start_year: int, log=print, download=_download) -> pd.DataFrame:
    """All open-market insider purchases filed since start_year (cached per quarter)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    qs = quarters(start_year)
    todo = [q for q in qs if not (CACHE / f"purchases_{q}.csv").is_file()]
    if todo:
        log(f"Downloading {len(todo)} quarters of SEC insider filings (each 10-60 MB; the first time "
            f"takes a while, then only new quarters are fetched)...")
        ua = user_agent()
    for n, q in enumerate(todo, 1):
        try:
            data = download(q, ua)
        except Exception as e:
            log(f"  {q}: download failed ({e}); it will be retried next run")
            continue
        if data is None:
            log(f"  {q}: not published by the SEC yet")
            continue
        df = parse_quarter(data)
        df.to_csv(CACHE / f"purchases_{q}.csv", index=False, date_format="%Y-%m-%d")
        log(f"  {q}: {len(df)} purchases ({n}/{len(todo)})")
        time.sleep(0.5)                                  # stay well under the SEC's rate limit
    frames = [pd.read_csv(CACHE / f"purchases_{q}.csv", parse_dates=["filing_date", "trans_date"],
                          dtype={"ticker": str, "issuer_cik": str, "owner_cik": str})
              for q in qs if (CACHE / f"purchases_{q}.csv").is_file()]
    if not frames:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates(["accession", "trans_date", "shares", "price"])
