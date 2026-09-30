import pandas as pd

from desk import market_data

from .conftest import make_bars


def fake_yahoo(available, calls):
    def download(syms, period, **kw):
        calls.append((tuple(syms), period))
        frames = {s: make_bars(n=300 if period == "2y" else 22, seed=len(s)) for s in syms if s in available}
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1)  # MultiIndex (ticker, field) like group_by="ticker"
    return download


def test_download_caches_and_reuses(tmp_path, monkeypatch):
    monkeypatch.setattr(market_data, "BAR_CACHE", tmp_path)
    monkeypatch.setattr(market_data.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(market_data.yf, "download", fake_yahoo({"SPY", "AAPL"}, calls))
    bars = market_data.download_bars(["AAPL", "MISSING"], priority=["SPY"], batch=2, log=lambda *a: None)
    assert set(bars) == {"SPY", "AAPL"}
    assert calls[0][0] == ("SPY", "AAPL")          # priority symbols first
    assert (tmp_path / "SPY.csv").is_file()

    calls.clear()
    again = market_data.download_bars(["AAPL"], priority=["SPY"], log=lambda *a: None)
    assert calls == [] and set(again) == {"SPY", "AAPL"}   # fresh cache: no download


def test_stale_cache_only_fetches_recent_month(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(market_data, "BAR_CACHE", tmp_path)
    monkeypatch.setattr(market_data.time, "sleep", lambda s: None)
    market_data._save_cached("AAPL", make_bars(end="2026-09-01"))
    os.utime(tmp_path / "AAPL.csv", (0, 0))                 # make the cache old
    calls = []
    monkeypatch.setattr(market_data.yf, "download", fake_yahoo({"AAPL"}, calls))
    bars = market_data.download_bars(["AAPL"], log=lambda *a: None)
    assert calls == [(("AAPL",), "1mo")]
    assert len(bars["AAPL"]) > 300


def test_failed_update_falls_back_to_cache(tmp_path, monkeypatch):
    import os
    monkeypatch.setattr(market_data, "BAR_CACHE", tmp_path)
    monkeypatch.setattr(market_data.time, "sleep", lambda s: None)
    market_data._save_cached("AAPL", make_bars())
    os.utime(tmp_path / "AAPL.csv", (0, 0))
    monkeypatch.setattr(market_data.yf, "download", fake_yahoo(set(), []))
    assert "AAPL" in market_data.download_bars(["AAPL"], log=lambda *a: None)
