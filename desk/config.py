"""Loads config/desk.toml into typed settings."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "desk.toml"
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


def load_settings(path: Path = DEFAULT_CONFIG) -> Settings:
    raw = tomllib.loads(Path(path).read_text())
    missing = [r for r in ROLES if r not in raw["models"] or r not in raw["prompts"]]
    if missing:
        raise ValueError(f"config is missing model or prompt version for: {missing}")
    return Settings(
        paper_equity=float(raw["account"]["paper_equity"]),
        allowlist=tuple(s.upper() for s in raw["universe"]["allowlist"]),
        regime_symbols=tuple(raw["universe"]["regime_symbols"]),
        sector_etfs=tuple(raw["universe"]["sector_etfs"]),
        risk=RiskLimits(**raw["risk"]),
        models=dict(raw["models"]),
        effort=dict(raw.get("effort", {})),
        prompt_versions=dict(raw["prompts"]),
    )
