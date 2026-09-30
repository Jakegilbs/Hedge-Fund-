"""Loads config/desk.toml into typed settings."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

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
