"""
Rate Limiter for API Calls
Prevents exceeding API rate limits for Alpaca, yfinance, and other data sources
"""

import time
from collections import deque
from datetime import datetime, timedelta
from typing import Optional


class RateLimiter:
    """
    Token bucket rate limiter
    Ensures we don't exceed API rate limits
    """

    def __init__(self, max_calls: int, time_window: int):
        """
        Initialize rate limiter

        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()

    def wait_if_needed(self):
        """
        Wait if we would exceed rate limit
        """
        now = time.time()

        # Remove calls outside the time window
        while self.calls and self.calls[0] < now - self.time_window:
            self.calls.popleft()

        # If we're at the limit, wait
        if len(self.calls) >= self.max_calls:
            sleep_time = self.time_window - (now - self.calls[0]) + 0.1
            if sleep_time > 0:
                print(f"⏳ Rate limit reached, waiting {sleep_time:.1f}s...")
                time.sleep(sleep_time)
                # Clean up old calls after waiting
                now = time.time()
                while self.calls and self.calls[0] < now - self.time_window:
                    self.calls.popleft()

        # Record this call
        self.calls.append(time.time())


class APIRateLimiters:
    """
    Centralized rate limiters for different APIs
    """

    def __init__(self):
        # Alpaca: 200 requests per minute (to be safe, use 180)
        self.alpaca_limiter = RateLimiter(max_calls=180, time_window=60)

        # yfinance: No official limit, but be conservative (2 per second)
        self.yfinance_limiter = RateLimiter(max_calls=120, time_window=60)

        # Interactive Brokers: varies by account type (conservative default)
        self.ibkr_limiter = RateLimiter(max_calls=50, time_window=1)

    def wait_for_alpaca(self):
        """Wait if needed before Alpaca API call"""
        self.alpaca_limiter.wait_if_needed()

    def wait_for_yfinance(self):
        """Wait if needed before yfinance API call"""
        self.yfinance_limiter.wait_if_needed()

    def wait_for_ibkr(self):
        """Wait if needed before IBKR API call"""
        self.ibkr_limiter.wait_if_needed()


def retry_with_backoff(func, max_retries: int = 3,
                       initial_delay: float = 1.0,
                       backoff_factor: float = 2.0):
    """
    Retry a function with exponential backoff

    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for each retry

    Returns:
        Result of function call or raises last exception
    """
    delay = initial_delay
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e

            # Check if it's a rate limit error
            error_msg = str(e).lower()
            is_rate_limit = any(phrase in error_msg for phrase in [
                'rate limit', 'too many requests', '429', 'quota exceeded'
            ])

            if attempt < max_retries:
                # Increase delay if it's a rate limit error
                actual_delay = delay * 3 if is_rate_limit else delay
                print(f"⚠️  API call failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
                print(f"   Retrying in {actual_delay:.1f}s...")
                time.sleep(actual_delay)
                delay *= backoff_factor
            else:
                # Last attempt failed
                if is_rate_limit:
                    print(f"❌ Rate limit exceeded after {max_retries + 1} attempts")
                break

    raise last_exception


# Global rate limiters instance
api_rate_limiters = APIRateLimiters()
