import os

from desk.config import load_env_file


def test_env_file_loads_without_overriding_terminal(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('﻿# comment\nANTHROPIC_API_KEY="sk-ant-file"\nOTHER=1\n', encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OTHER", "terminal")
    load_env_file(env)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-file"
    assert os.environ["OTHER"] == "terminal"


def test_missing_env_file_is_fine(tmp_path):
    load_env_file(tmp_path / "nope")


def test_utf16_env_file_from_powershell(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("ANTHROPIC_API_KEY=sk-ant-ps\n", encoding="utf-16")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    load_env_file(env)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-ps"
