"""Versioned prompt files: prompts/<role>/<version>.md with {{placeholders}}."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass

from .config import ROOT

PROMPTS_DIR = ROOT / "prompts"
PLACEHOLDER = re.compile(r"\{\{\s*([a-z_][a-z0-9_]*)\s*\}\}")


@dataclass(frozen=True)
class Prompt:
    role: str
    version: str
    template: str

    @property
    def fingerprint(self) -> str:
        """Short hash of the template, logged with every run to tie results to exact wording."""
        return hashlib.sha256(self.template.encode()).hexdigest()[:12]

    def placeholders(self) -> set[str]:
        return set(PLACEHOLDER.findall(self.template))

    def render(self, **values: object) -> str:
        missing = self.placeholders() - values.keys()
        if missing:
            raise KeyError(f"{self.role}/{self.version} needs values for: {sorted(missing)}")

        def fill(m: re.Match) -> str:
            v = values[m.group(1)]
            return v if isinstance(v, str) else json.dumps(v, indent=1, default=str)

        return PLACEHOLDER.sub(fill, self.template)


def load_prompt(role: str, version: str) -> Prompt:
    path = PROMPTS_DIR / role / f"{version}.md"
    if not path.is_file():
        raise FileNotFoundError(f"no prompt file at {path}")
    return Prompt(role=role, version=version, template=path.read_text())
