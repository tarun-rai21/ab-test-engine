"""
api/security.py — optional API-key authentication.

Set the API_KEY environment variable to require every data endpoint to
receive the same value in an `X-API-Key` header; leave it unset for local
development, where the API stays open. The key is read on each request, so
it can be rotated by restarting with a new value, and compared in constant
time (hmac.compare_digest) so response timing doesn't reveal how much of a
guessed key matched.
"""

from __future__ import annotations

import hmac
import os

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader

API_KEY_ENV = "API_KEY"

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(provided: str | None = Security(_api_key_header)) -> None:
    expected = os.environ.get(API_KEY_ENV)
    if not expected:
        return  # authentication disabled (local development)
    if provided is None or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API key (send it in the X-API-Key header).",
            headers={"WWW-Authenticate": "APIKey"},
        )
