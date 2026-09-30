"""One call to Claude per agent, with a strict report format and cost tracking."""
from __future__ import annotations

from dataclasses import dataclass, field

import anthropic
from pydantic import BaseModel

from .prompts import Prompt

# USD per million tokens (input, output), Anthropic first-party pricing.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
# Models that accept the effort setting and server-side refusal fallbacks.
MODERN_MODELS = {"claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"


@dataclass
class AgentResult:
    role: str
    prompt_version: str
    prompt_fingerprint: str
    model_requested: str
    model_served: str | None = None
    report: BaseModel | None = None
    error: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    extra: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.report is not None and self.error is None


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = PRICES.get(model, PRICES["claude-opus-5-5"])  # unknown model: price as Opus
    return round((input_tokens * inp + output_tokens * out) / 1_000_000, 6)


class ClaudeRunner:
    """Runs one agent: rendered prompt in, validated report out. Never raises for API errors."""

    def __init__(self, client: anthropic.Anthropic | None = None):
        self.client = client or anthropic.Anthropic()

    def run(self, prompt: Prompt, rendered: str, schema: type[BaseModel], model: str,
            effort: str | None = None, max_tokens: int = 16000) -> AgentResult:
        result = AgentResult(role=prompt.role, prompt_version=prompt.version,
                             prompt_fingerprint=prompt.fingerprint, model_requested=model)
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": rendered}],
            "output_format": schema,
        }
        if model in MODERN_MODELS:
            # Re-run on another model if this one declines, instead of returning nothing.
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
            if effort:
                kwargs["output_config"] = {"effort": effort}
        try:
            response = self.client.beta.messages.parse(**kwargs)
        except anthropic.APIConnectionError as e:
            result.error = f"network error: {e}"
            return result
        except anthropic.APIStatusError as e:
            result.error = f"API error {e.status_code}: {e.message}"
            return result
        except Exception as e:  # includes schema validation failures of the reply
            result.error = f"invalid reply: {e}"
            return result

        result.model_served = response.model
        result.input_tokens = response.usage.input_tokens
        result.output_tokens = response.usage.output_tokens
        result.cost_usd = cost_usd(response.model, result.input_tokens, result.output_tokens)
        if response.stop_reason == "refusal":
            result.error = "model declined to answer"
        elif response.stop_reason == "max_tokens":
            result.error = "reply cut off at max_tokens"
        elif response.parsed_output is None:
            result.error = "reply did not match the report format"
        else:
            result.report = response.parsed_output
        return result
