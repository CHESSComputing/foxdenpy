"""Configuration handling for the foxden client.

Configuration is resolved by layering, lowest to highest precedence:

    built-in defaults  ->  ~/.foxden.yaml  ->  environment variables  ->  explicit args

Environment variables:
    FOXDEN_URL          frontend base URL, e.g. http://localhost:8344
    FOXDEN_STORAGE_URL  storage service base URL (defaults to FOXDEN_URL)
    FOXDEN_TOKEN        bearer token used for the Authorization header
    FOXDEN_VERIFY_SSL   "0"/"false" to disable TLS verification
    FOXDEN_TIMEOUT      per-request timeout in seconds
    FOXDEN_CACHE_DIR    optional local directory to cache downloaded objects

If FOXDEN_TOKEN is not set, a token is looked up in ~/.foxden/token as a
fallback (a plain text file containing just the token), matching how
other FOXDEN CLI tools store credentials.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:  # pyyaml is optional; config files are optional too
    yaml = None

DEFAULT_FRONTEND_URL = "http://localhost:8344"
DEFAULT_CONFIG_PATH = Path.home() / ".foxden.yaml"
DEFAULT_TOKEN_PATH = Path.home() / ".foxden" / "token"


@dataclass
class FoxdenConfig:
    url: str = DEFAULT_FRONTEND_URL
    storage_url: Optional[str] = None
    token: Optional[str] = None
    verify_ssl: bool = True
    timeout: float = 30.0
    connect_timeout: float = 10.0
    retries: int = 3
    cache_dir: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.storage_url:
            self.storage_url = self.url

    @classmethod
    def load(
        cls,
        url: Optional[str] = None,
        token: Optional[str] = None,
        config_path: Optional[str] = None,
    ) -> "FoxdenConfig":
        cfg: dict = {}

        path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        if path.exists() and yaml is not None:
            with open(path) as f:
                cfg.update(yaml.safe_load(f) or {})

        cfg["url"] = os.environ.get("FOXDEN_URL", cfg.get("url", DEFAULT_FRONTEND_URL))
        cfg["storage_url"] = os.environ.get("FOXDEN_STORAGE_URL", cfg.get("storage_url"))
        cfg["verify_ssl"] = _as_bool(os.environ.get("FOXDEN_VERIFY_SSL", cfg.get("verify_ssl", True)))
        cfg["timeout"] = float(os.environ.get("FOXDEN_TIMEOUT", cfg.get("timeout", 30.0)))
        cfg["connect_timeout"] = float(
            os.environ.get("FOXDEN_CONNECT_TIMEOUT", cfg.get("connect_timeout", 10.0))
        )
        cfg["retries"] = int(os.environ.get("FOXDEN_RETRIES", cfg.get("retries", 3)))
        cfg["cache_dir"] = os.environ.get("FOXDEN_CACHE_DIR", cfg.get("cache_dir"))

        env_token = os.environ.get("FOXDEN_TOKEN")
        if env_token:
            cfg["token"] = env_token
        elif not cfg.get("token"):
            cfg["token"] = _read_token_file()

        if url:
            cfg["url"] = url
        if token:
            cfg["token"] = token

        return cls(**cfg)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("0", "false", "no", "")


def _read_token_file() -> Optional[str]:
    if DEFAULT_TOKEN_PATH.exists():
        return DEFAULT_TOKEN_PATH.read_text().strip()
    return None
