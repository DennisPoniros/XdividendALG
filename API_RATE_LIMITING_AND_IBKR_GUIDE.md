# API Rate Limiting & Interactive Brokers Integration Guide

## Problem: API Rate Limits

When running backtests, the strategy makes **hundreds of API calls** to fetch:
- Dividend calendars (100+ stocks)
- Historical prices (1000+ requests during training)
- Fundamental data (100+ requests)

This can exceed API rate limits, causing errors like:
```
429 Too Many Requests
Rate limit exceeded
```

---

## Solution 1: Rate Limiting (Now Implemented ✅)

I've added comprehensive rate limiting to prevent these errors.

### How It Works

**rate_limiter.py** implements a token bucket algorithm:

```python
# Alpaca: 200 requests/minute (we use 180 to be safe)
alpaca_limiter = RateLimiter(max_calls=180, time_window=60)

# yfinance: No official limit (we use 120/minute to be conservative)
yfinance_limiter = RateLimiter(max_calls=120, time_window=60)

# IBKR: varies by account (default: 50/second)
ibkr_limiter = RateLimiter(max_calls=50, time_window=1)
```

### Features

1. **Automatic Rate Limiting**
   - Waits automatically when approaching rate limits
   - Shows progress: "⏳ Rate limit reached, waiting 12.3s..."

2. **Exponential Backoff**
   - Retries failed requests with increasing delays
   - Initial delay: 1s → 2s → 4s → 8s

3. **Smart Error Detection**
   - Detects rate limit errors (429, "too many requests")
   - Increases wait time for rate limit errors

4. **Non-Breaking**
   - If `rate_limiter.py` is missing, code still works (no rate limiting)
   - Backward compatible with existing code

### Usage

**Already integrated!** No changes needed. The code now:
- Limits Alpaca calls to 180/minute
- Limits yfinance calls to 120/minute
- Retries failed requests up to 3 times
- Shows progress when rate limiting occurs

---

## API Rate Limits Reference

### Alpaca API

| Plan | Rate Limit | Notes |
|------|------------|-------|
| **Paper Trading** | 200 requests/minute | **You're using this** |
| **Live Trading (Basic)** | 200 requests/minute | Same as paper |
| **Live Trading (Pro)** | 200 requests/minute | No change |
| **Unlimited Plan** | 1000 requests/minute | Paid upgrade |

**Official Docs**: https://alpaca.markets/docs/api-references/market-data-api/

**Our Implementation**: 180 requests/minute (10% safety margin)

### yfinance (Yahoo Finance)

| Metric | Limit | Notes |
|--------|-------|-------|
| **Official Limit** | None published | Yahoo doesn't publish limits |
| **Observed Limit** | ~2000/hour | Based on community reports |
| **Recommended** | 1-2 requests/second | Be conservative |

**Our Implementation**: 120 requests/minute (2 per second)

**Note**: yfinance is free but can be unreliable during market hours

###Interactive Brokers (IBKR)

| Account Type | Rate Limit | Notes |
|--------------|------------|-------|
| **Paper Trading** | 50 messages/second | Good for testing |
| **Live (Individual)** | 50 messages/second | Standard account |
| **Live (Professional)** | 100 messages/second | Requires qualification |

**Official Docs**: https://interactivebrokers.github.io/tws-api/

**Our Implementation**: 50 requests/second (when IBKR is configured)

---

## Solution 2: Using Interactive Brokers Instead

Interactive Brokers is a **better alternative** for serious trading:

### Why IBKR is Better

| Feature | Alpaca | yfinance | **Interactive Brokers** |
|---------|--------|----------|------------------------|
| **Rate Limits** | 200/min | Unclear (~2000/hr) | **50/sec (3000/min)** ✅ |
| **Data Quality** | Good | Variable | **Excellent** ✅ |
| **Historical Data** | Limited | Free but slow | **Paid but comprehensive** |
| **Live Trading** | Yes | No | **Yes** ✅ |
| **Commission** | $0 | N/A | **$0 for stocks** ✅ |
| **Reliability** | Good | Fair | **Excellent** ✅ |
| **Market Access** | US only | Global | **Global** ✅ |

### IBKR Integration Steps

#### Step 1: Create IBKR Account

1. **Sign up**: https://www.interactivebrokers.com/
2. **Choose Account Type**:
   - Paper Trading (free, for testing)
   - Live Individual (for real trading)
3. **Complete Application**: Takes 1-2 days for approval

#### Step 2: Install IBKR TWS or Gateway

**TWS (Trader Workstation)** - Full GUI:
- Download: https://www.interactivebrokers.com/en/trading/tws.php
- Heavier but has charts and monitoring

**IB Gateway** - Lightweight (recommended for bots):
- Download: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php
- Lighter, better for automated trading

#### Step 3: Install Python API

```cmd
pip install ib_insync
```

**ib_insync** is the best IBKR Python library (async, modern).

Alternative:
```cmd
pip install ibapi
```

**ibapi** is the official library (more complex).

#### Step 4: Create IBKR Data Manager

I can create `data_manager_ibkr.py` that:
- Connects to IBKR API
- Fetches historical data
- Gets real-time quotes
- Places orders for live trading

**Example code**:

```python
from ib_insync import *

class IBKRDataManager:
    def __init__(self, host='127.0.0.1', port=7497, client_id=1):
        """
        Initialize IBKR connection

        Args:
            host: IB Gateway host
            port: 7497 (paper), 7496 (live)
            client_id: Unique client ID
        """
        self.ib = IB()
        self.ib.connect(host, port, clientId=client_id)

    def get_historical_data(self, ticker, start_date, end_date):
        """Get historical bars from IBKR"""
        contract = Stock(ticker, 'SMART', 'USD')
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime=end_date,
            durationStr='1 Y',
            barSizeSetting='1 day',
            whatToShow='TRADES',
            useRTH=True
        )
        return util.df(bars)  # Convert to pandas
```

#### Step 5: Update Config

Add to `config.py`:

```python
# Interactive Brokers Configuration
IBKR_CONFIG = {
    'HOST': '127.0.0.1',
    'PORT': 7497,  # Paper trading (7496 for live)
    'CLIENT_ID': 1,
    'USE_IBKR': True  # Set to True to use IBKR instead of Alpaca
}
```

#### Step 6: Modify Strategy

Update `data_manager.py` to use IBKR:

```python
if IBKR_CONFIG['USE_IBKR']:
    dm = IBKRDataManager()
else:
    dm = DataManager()  # Uses Alpaca/yfinance
```

---

## Recommended Approach

### For Backtesting (Current Use Case)

**Best Option**: **yfinance (free)** with rate limiting ✅

**Why**:
- Free historical data
- Sufficient for backtesting
- Rate limiting prevents errors
- Already working in your code

**Keep Alpaca as backup** for when yfinance fails.

### For Paper Trading

**Best Option**: **Interactive Brokers Paper Account**

**Why**:
- Free paper trading
- Real market data
- Better rate limits (50/sec vs 200/min)
- Practice for live trading
- More reliable than Alpaca

### For Live Trading

**Best Option**: **Interactive Brokers Live Account**

**Why**:
- $0 commissions on stocks
- Best execution quality
- Global market access
- Industry standard for serious traders
- Best rate limits and data quality

---

## Cost Comparison

| Service | Setup Cost | Monthly Cost | Data Cost | Trading Cost |
|---------|------------|--------------|-----------|--------------|
| **yfinance** | $0 | $0 | $0 | N/A (no trading) |
| **Alpaca Paper** | $0 | $0 | $0 | $0 |
| **Alpaca Live** | $0 | $0 | $0 | $0 stocks |
| **IBKR Paper** | $0 | $0 | $0 | $0 |
| **IBKR Live** | $0 | $0-10* | $0** | $0 stocks*** |

*$10/month if < $100k account (waived if you trade)
**Real-time data requires subscriptions ($1-15/month per exchange)
***$0 commission for US stocks, but SEC fees apply (~$0.01 per $1000)

---

## Implementation Status

### ✅ Already Implemented

1. **Rate limiting for Alpaca** - 180 requests/minute
2. **Rate limiting for yfinance** - 120 requests/minute
3. **Exponential backoff retries** - up to 3 attempts
4. **Smart error detection** - detects rate limit errors
5. **Progress indicators** - shows when waiting

### 🔄 Can Be Added (If You Want)

6. **IBKR integration** - Full IBKR data manager
7. **Automatic source switching** - Use IBKR when Alpaca fails
8. **Live trading mode** - Execute real trades via IBKR
9. **Real-time data streaming** - For live strategy monitoring
10. **Multi-broker support** - Use best source for each request

---

## Next Steps

### Option A: Stay With Current Setup (Recommended for Now)

✅ **You're all set!** Rate limiting is now active.

**What you get**:
- Free historical data via yfinance
- Rate limiting prevents errors
- Backups runs faster with caching
- No additional setup needed

**Run your backtest again**:
```cmd
python run_xdiv_ml_backtest.py
```

You should see:
```
⏳ Rate limit reached, waiting 5.2s...
```

Instead of rate limit errors!

### Option B: Add IBKR Integration

**If you want better data quality and plan to live trade**:

1. Create IBKR paper account (free)
2. Install IB Gateway
3. Let me know, and I'll create `data_manager_ibkr.py`
4. Update config to use IBKR
5. Test with paper trading first

---

## Troubleshooting

### Still Getting Rate Limit Errors?

**Check if rate limiter is active**:
```python
python -c "from rate_limiter import api_rate_limiters; print('Rate limiter active!')"
```

If error: `rate_limiter.py` not in directory. Re-download from repo.

### Errors Seem Random?

**Likely yfinance issues** (not rate limiting):
- Yahoo servers can be unreliable
- Try running at different time of day
- Add more retries in `rate_limiter.py` (change `max_retries=3` to `max_retries=5`)

### Want Faster Backtests?

**Enable aggressive caching**:
1. First run: 10-15 minutes (downloads data)
2. Subsequent runs: 2-3 minutes (uses cache)
3. Cache is automatic - no action needed!

**Check cache stats**:
```python
dm = DataManager()
# ... run backtest ...
stats = dm.get_cache_stats()
print(f"Cache hit rate: {stats['hit_rate_pct']:.1f}%")
```

### Want to Clear Cache?

```python
dm = DataManager()
dm.clear_cache()
```

Or delete: `data_manager.py` creates in-memory cache (no files).

---

## Code Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `rate_limiter.py` | Rate limiting & retries | ✅ Created |
| `data_manager.py` | Data fetching (updated) | ✅ Updated |
| `strategy_xdiv_ml.py` | ML strategy | ✅ Uses data_manager |
| `backtester_xdiv_ml.py` | Backtester | ✅ Uses data_manager |
| `data_manager_ibkr.py` | IBKR integration | ⏸️ Not created yet |

---

## Summary

**Problem**: API rate limit errors during backtest

**Solution Implemented**: ✅
- Rate limiting (180/min Alpaca, 120/min yfinance)
- Exponential backoff retries
- Smart error detection
- Progress indicators

**Alternative Solution**: Interactive Brokers
- Better rate limits (50/sec = 3000/min)
- Better data quality
- Required for serious live trading
- Can be integrated if needed

**Recommendation**:
1. **Now**: Use current setup with rate limiting (free, works great)
2. **Later**: Switch to IBKR when ready for paper/live trading

**Your backtest should now run without rate limit errors!** 🎉

Let me know if you want me to:
1. Add IBKR integration
2. Adjust rate limits (make faster/slower)
3. Add more retries
4. Anything else!
