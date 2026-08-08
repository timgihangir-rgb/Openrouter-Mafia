"""Configuration loaded from environment variables. No secrets are hardcoded."""

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    models: list[str] = field(default_factory=lambda: _parse_models())
    num_players: int = field(default_factory=lambda: int(os.environ.get("NUM_PLAYERS", "7")))
    role_distribution: dict = field(
        default_factory=lambda: {"Mafia": 2, "Detective": 1, "Doctor": 1, "Civilian": 3}
    )
    max_retries: int = 3
    timeout: float = 30.0
    discussion_rounds: int = field(
        default_factory=lambda: int(os.environ.get("DISCUSSION_ROUNDS", "2"))
    )
    http_referer: str = "https://github.com/mafia-ai-game"
    app_title: str = "Mafia AI Game"
    max_tokens: int = 4096
    reasoning_enabled: bool = field(
        default_factory=lambda: os.environ.get("REASONING", "1") in ("1", "true", "True")
    )

    def __post_init__(self):
        if not self.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable is not set. "
                "Copy .env.example to .env and fill in your key, "
                "or export OPENROUTER_API_KEY in your shell."
            )
        if self.num_players > len(self.models):
            raise ValueError(
                f"Need at least {self.num_players} models, got {len(self.models)}"
            )


def _parse_models() -> list[str]:
    """Parse models from the MODELS env var or fall back to defaults."""
    env_models = os.environ.get("MODELS")
    if env_models:
        return [m.strip() for m in env_models.split(",") if m.strip()]
    return [
        "poolside/laguna-s-2.1",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "google/gemma-4-31b-it",
        "deepseek/deepseek-v4-flash-0731",
        "qwen/qwen3.7-flash",
        "google/gemini-3.5-flash-lite",
        "mistralai/mistral-medium-3",
    ]
