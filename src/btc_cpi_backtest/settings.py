"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Define runtime configuration for data providers and persistence."""

    ccxt_api_key: Optional[str] = Field(default=None, env="CCXT_API_KEY")
    ccxt_api_secret: Optional[str] = Field(default=None, env="CCXT_API_SECRET")
    ccxt_password: Optional[str] = Field(default=None, env="CCXT_PASSWORD")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings."""

    return Settings()
