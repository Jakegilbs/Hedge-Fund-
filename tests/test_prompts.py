import pytest

from desk.config import ROLES, load_settings
from desk.prompts import load_prompt


def test_every_configured_prompt_exists():
    s = load_settings()
    for role in ROLES:
        assert load_prompt(role, s.prompt_versions[role]).template


def test_render_fails_loudly_on_missing_values():
    p = load_prompt("technical_analyst", "v1")
    with pytest.raises(KeyError):
        p.render()


def test_render_fills_every_placeholder():
    p = load_prompt("regime_analyst", "v1")
    out = p.render(regime_data={"SPY": 1})
    assert "{{" not in out and '"SPY": 1' in out


def test_fingerprint_changes_with_wording():
    p = load_prompt("technical_analyst", "v1")
    q = type(p)(role=p.role, version=p.version, template=p.template + " ")
    assert p.fingerprint != q.fingerprint


def test_config_has_sane_limits():
    s = load_settings()
    assert 0 < s.risk.risk_per_trade <= 0.02
    assert s.risk.min_reward_risk >= 1.5
    assert "SPY" in s.allowlist
