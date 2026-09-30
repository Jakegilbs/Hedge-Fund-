"""Loads config/desk.toml into typed settings."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from .options import OptionsConfig

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "desk.toml"
ENV_FILE = ROOT / ".env"
DATA_DIR = ROOT / "data"  # everything the desk saves stays inside the project folder
ROLES = ("technical_analyst", "catalyst_analyst", "regime_analyst", "portfolio_manager")


@dataclass(frozen=True)
class RiskLimits:
    risk_per_trade: float
    max_position_pct: float
    max_open_positions: int
    min_reward_risk: float
    daily_halt_pct: float
    weekly_halt_pct: float
    max_drawdown_pct: float
    max_stop_distance_pct: float = 1.0   # 1.0 = no cap
    min_conviction: int = 1

    @property
    def all_in(self) -> bool:
        return self.max_open_positions == 1 and self.max_position_pct >= 1.0


@dataclass(frozen=True)
class Settings:
    paper_equity: float
    allowlist: tuple[str, ...]
    regime_symbols: tuple[str, ...]
    sector_etfs: tuple[str, ...]
    risk: RiskLimits
    models: dict[str, str]
    effort: dict[str, str]
    prompt_versions: dict[str, str]
    include_sp500: bool = False
    max_candidates: int = 8
    min_price: float = 10.0
    min_dollar_volume: float = 50_000_000
    instrument: str = "stock"            # "stock" or "options" (long calls and puts only)
    options_extra: tuple[str, ...] = ()
    options_min_price: float = 5.0
    options: "OptionsConfig" = None      # set by load_settings

    @property
    def scan_min_price(self) -> float:
        return self.options_min_price if self.instrument == "options" else self.min_price

    @property
    def etfs(self) -> set[str]:
        """Funds on the allowlist or used for context: they never have earnings."""
        known = {"SPY", "QQQ", "IWM", "DIA", "TLT"}
        return known | set(self.sector_etfs) | {s for s in self.regime_symbols if not s.startswith("^")}


def load_settings(path: Path = DEFAULT_CONFIG) -> Settings:
    raw = tomllib.loads(Path(path).read_text())
    missing = [r for r in ROLES if r not in raw["models"] or r not in raw["prompts"]]
    if missing:
        raise ValueError(f"config is missing model or prompt version for: {missing}")
    if raw.get("strategy", {}).get("instrument") == "options" and "portfolio_manager_options" not in raw["prompts"]:
        raise ValueError("options mode needs prompts.portfolio_manager_options")
    if raw.get("strategy", {}).get("instrument", "stock") not in ("stock", "options"):
        raise ValueError('strategy.instrument must be "stock" or "options"')
    return Settings(
        paper_equity=float(raw["account"]["paper_equity"]),
        allowlist=tuple(s.upper() for s in raw["universe"]["allowlist"]),
        regime_symbols=tuple(raw["universe"]["regime_symbols"]),
        sector_etfs=tuple(raw["universe"]["sector_etfs"]),
        risk=RiskLimits(**raw["risk"]),
        models=dict(raw["models"]),
        effort=dict(raw.get("effort", {})),
        prompt_versions=dict(raw["prompts"]),
        include_sp500=bool(raw["universe"].get("include_sp500", False)),
        max_candidates=int(raw["universe"].get("max_candidates", 8)),
        min_price=float(raw["universe"].get("min_price", 10.0)),
        min_dollar_volume=float(raw["universe"].get("min_dollar_volume", 50_000_000)),
        instrument=str(raw.get("strategy", {}).get("instrument", "stock")),
        options_extra=tuple(t.upper() for t in raw["universe"].get("options_extra", [])),
        options_min_price=float(raw["universe"].get("options_min_price", 5.0)),
        options=OptionsConfig(**raw.get("options", {})),
    )


def load_env_file(path: Path = ENV_FILE) -> None:
    """Load KEY=value lines from the project's .env file (git-ignored) into the environment.

    Values already set in the terminal win. Lines starting with # are ignored.
    """
    if not path.is_file():
        return
    raw = path.read_bytes()
    # PowerShell's `echo x > .env` writes UTF-16; accept it as well as UTF-8.
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
