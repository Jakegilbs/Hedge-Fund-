import io
import zipfile

import numpy as np
import pandas as pd
import pytest

from desk import insider_study as st
from desk import insiders as ins


def sec_zip() -> bytes:
    """A tiny quarterly zip in the SEC's Insider Transactions Data Set layout."""
    sub = ("ACCESSION_NUMBER\tFILING_DATE\tPERIOD_OF_REPORT\tDOCUMENT_TYPE\tISSUERCIK\tISSUERNAME\tISSUERTRADINGSYMBOL\n"
           "A1\t05-MAR-2024\t04-MAR-2024\t4\t111\tAcme Corp\tacme\n"
           "A2\t06-MAR-2024\t05-MAR-2024\t4\t111\tAcme Corp\tACME\n"
           "A3\t06-MAR-2024\t05-MAR-2024\t4/A\t111\tAcme Corp\tACME\n"
           "A4\t07-MAR-2024\t06-MAR-2024\t4\t222\tBeta Inc\tBRK.B\n")
    own = ("ACCESSION_NUMBER\tRPTOWNERCIK\tRPTOWNERNAME\tRPTOWNER_RELATIONSHIP\tRPTOWNER_TITLE\n"
           "A1\t901\tSmith Jane\tDirector,Officer\tChief Executive Officer\n"
           "A1\t902\tSmith Trust\tTenPercentOwner\t\n"
           "A2\t903\tDoe John\tDirector\t\n"
           "A3\t903\tDoe John\tDirector\t\n"
           "A4\t904\tRoe Ann\tOfficer\tCFO\n")
    trans = ("ACCESSION_NUMBER\tNONDERIV_TRANS_SK\tSECURITY_TITLE\tTRANS_DATE\tTRANS_CODE\tTRANS_SHARES\t"
             "TRANS_PRICEPERSHARE\tTRANS_ACQUIRED_DISP_CD\tSHRS_OWND_FOLWNG_TRANS\tDIRECT_INDIRECT_OWNERSHIP\n"
             "A1\t1\tCommon\t04-MAR-2024\tP\t1,000\t10.50\tA\t5000\tD\n"
             "A1\t2\tCommon\t04-MAR-2024\tS\t500\t10.60\tD\t4500\tD\n"          # a sale: ignored
             "A2\t3\tCommon\t05-MAR-2024\tP\t200\t11\tA\t800\tI\n"
             "A2\t4\tCommon\t05-MAR-2024\tA\t300\t0\tA\t1100\tD\n"              # a grant: ignored
             "A3\t5\tCommon\t05-MAR-2024\tP\t200\t11\tA\t800\tI\n"              # amendment: ignored
             "A4\t6\tCommon\t06-MAR-2024\tP\t100\t400\tA\t900\tD\n")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("2024q1_form345/SUBMISSION.tsv", sub)
        z.writestr("2024q1_form345/REPORTINGOWNER.tsv", own)
        z.writestr("2024q1_form345/NONDERIV_TRANS.tsv", trans)
    return buf.getvalue()


def test_parse_keeps_only_open_market_purchases_on_form_4():
    df = ins.parse_quarter(sec_zip())
    assert list(df.accession) == ["A1", "A2", "A4"]
    a1 = df[df.accession == "A1"].iloc[0]
    assert a1.ticker == "ACME" and a1.shares == 1000 and a1.value == pytest.approx(10500)
    assert a1.filing_date == pd.Timestamp("2024-03-05") and a1.owner == "Smith Jane"      # one owner per filing
    assert df[df.accession == "A4"].iloc[0].ticker == "BRK-B"


def test_quarters_only_finished_ones():
    from datetime import date
    assert ins.quarters(2025, today=date(2025, 8, 1)) == ["2025q1", "2025q2"]


def test_load_purchases_caches_and_skips_unpublished(tmp_path, monkeypatch):
    monkeypatch.setattr(ins, "CACHE", tmp_path)
    monkeypatch.setenv("SEC_USER_AGENT", "Test test@example.com")
    monkeypatch.setattr(ins, "quarters", lambda y: ["2024q1", "2024q2"])
    monkeypatch.setattr(ins.time, "sleep", lambda s: None)
    calls = []
    fake = lambda q, ua: calls.append(q) or (sec_zip() if q == "2024q1" else None)
    df = ins.load_purchases(2024, log=lambda *a: None, download=fake)
    assert len(df) == 3 and calls == ["2024q1", "2024q2"]
    calls.clear()
    ins.load_purchases(2024, log=lambda *a: None, download=fake)
    assert calls == ["2024q2"]                        # cached quarter is not downloaded again


def test_user_agent_is_required(monkeypatch):
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    with pytest.raises(SystemExit):
        ins.user_agent()


def purchases(rows):
    return pd.DataFrame([{"ticker": t, "filing_date": pd.Timestamp(d), "owner": o, "owner_cik": o,
                          "relationship": rel, "title": title, "value": v, "shares": v / 10, "issuer": t}
                         for t, d, o, rel, title, v in rows])


def test_events_group_purchases_and_upgrade_to_clusters():
    p = purchases([("X", "2024-01-02", "a", "Director", "", 50e3),
                   ("X", "2024-01-05", "a", "Director", "", 10e3),       # same insider again: no new event
                   ("X", "2024-01-10", "b", "Officer", "CEO", 200e3),     # a 2nd insider: cluster event
                   ("X", "2024-03-01", "c", "Director", "", 20e3),        # > 30 days later: fresh event
                   ("Y", "2024-01-02", "d", "TenPercentOwner", "", 1e6)])
    ev = st.build_events(p)
    x = ev[ev.ticker == "X"]
    assert list(x.insiders) == [1, 2, 1] and list(x.date.dt.strftime("%m-%d")) == ["01-02", "01-10", "03-01"]
    assert x.iloc[1].value == pytest.approx(260e3) and x.iloc[1].who == "CEO/CFO/President bought"
    assert ev[ev.ticker == "Y"].iloc[0].who == "10% owners only"


def flat(n=300, price=10.0, vol=1e6, end="2024-06-28"):
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"Open": price, "High": price, "Low": price, "Close": price, "Volume": vol}, index=idx)


def test_returns_start_after_the_filing_and_use_a_size_matched_benchmark():
    stock, spy, iwm = flat(), flat(price=500, vol=1e8), flat(price=200, vol=1e8)
    idx = stock.index
    filed = idx[200]
    stock.loc[idx[201]:, ["Open", "Close"]] = 12.0                 # jumps the day after the filing
    stock.loc[idx[201 + 19]:, "Close"] = 13.2                       # +10% from the entry open by day 20
    iwm.loc[idx[201 + 19]:, "Close"] = 210.0                        # small caps +5%
    ev = pd.DataFrame([{"ticker": "S", "date": filed, "issuer": "S", "insiders": 2, "value": 1e5,
                        "purchases": 2, "who": "directors only", "avg_price": 10.0}])
    [row] = st.add_returns(ev, {"S": stock, "SPY": spy, "IWM": iwm}).to_dict("records")
    assert row["entry_date"] == idx[201].date() and row["entry_price"] == 12.0      # no buying on the old price
    assert row["benchmark"] == "IWM" and row["excess_20"] == pytest.approx(5.0)


def test_report_runs():
    rng = np.random.default_rng(0)
    n = 600
    ev = pd.DataFrame({"ticker": [f"T{i % 80}" for i in range(n)],
                       "date": pd.to_datetime(pd.bdate_range("2016-01-04", "2021-12-30")[rng.integers(0, 1500, n)]),
                       "insiders": rng.integers(1, 5, n), "value": rng.lognormal(12, 1.5, n),
                       "who": rng.choice(["CEO/CFO/President bought", "directors only", "10% owners only"], n),
                       "dollar_volume": rng.lognormal(16, 2, n), "prior_60d_pct": rng.normal(-5, 20, n)})
    for h in st.HORIZONS:
        ev[f"excess_{h}"] = rng.normal(0, 10, n)
    text = st.report(ev, locked=True, coverage="Coverage: test")
    for part in ("By how many insiders", "By who bought", "company size", "year by year", "2022+ locked"):
        assert part in text
