"""
backend/rate_limit.py

Shared rate limiter instance for all API routes.

All route files must import `limiter` from this module instead of creating
their own Limiter instances. This ensures:
  - A single storage backend (in-memory by default) tracks all rate limits
  - `default_limits` applies to routes without explicit @limiter.limit()
  - The exception handler in main.py handles RateLimitExceeded correctly
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
