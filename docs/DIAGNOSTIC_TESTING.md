# Diagnostic Testing Guide for Dividend Capture Algorithm

This guide helps you understand **why** stocks aren't generating trades and how to adjust filters.

## Understanding the Filtering Pipeline

```
Dividend Calendar (19 stocks in Oct 2024)
    ↓
[1] Fundamental Screening (market cap, ROE, debt, etc.)
    ↓
[2] Quality Score Filtering (min 50-70)
    ↓
[3] Technical Entry Filters (RSI, momentum, z-score, volatility)
    ↓
Entry Signals Generated
```

---

## Quick Diagnostic Checklist

**Pull latest code and run:**

```python
%cd /content/XdividendALG
!git pull origin main

# Reload
import importlib, sys
for mod in ['config', 'data_manager', 'strategy', 'backtester']:
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])

# Run with diagnostics
from backtester import Backtester

bt = Backtester(start_date='2024-10-01', end_date='2024-10-31', initial_capital=50_000)
results = bt.run_backtest(mode='full')
```

**You'll now see:**

```
🔍 Screening 14 dividend candidates...
✅ 5 stocks passed screening

🎯 Generating entry signals for 5 candidates...
✅ Generated 0 entry signals

📋 Entry filter failures:
   ABT: Z-score 1.23 outside range [-2.0, 0.0]
   PG: Negative momentum -1.45%
   GD: Days to ex-div 7 not in preferred window [3, 4, 5]
   PNC: RSI 72.3 outside range [30, 70]
```

---

## Common Failure Reasons & Fixes

### 1. **Z-Score Filter (MOST COMMON)**

**Problem:** Stock passes screening but z-score is positive (above mean)

```
ABT: Z-score 1.23 outside range [-2.0, 0.0]
```

**What it means:** Stock is trading ABOVE its 20-day mean - strategy wants oversold stocks

**Fix Options:**

```python
# Option A: Widen the range
from config import entry_config
entry_config.z_score_max = 1.0  # Was 0.0, now allows slightly overbought

# Option B: Disable z-score filter
entry_config.use_z_score_filter = False
```

---

### 2. **Negative Momentum**

**Problem:**
```
PG: Negative momentum -1.45%
```

**What it means:** Stock declined over last 20 days - strategy requires positive momentum

**Fix:**

```python
from config import entry_config
entry_config.require_positive_momentum = False  # Accept any momentum
```

---

### 3. **Days to Ex-Dividend Window**

**Problem:**
```
GD: Days to ex-div 7 not in preferred window [3, 4, 5]
```

**What it means:** Stock ex-dividend is 7 days away, but strategy only enters 3-5 days before

**Fix:**

```python
from config import entry_config
entry_config.preferred_entry_days = [3, 4, 5, 6, 7, 8]  # Wider window
```

---

### 4. **RSI Outside Range**

**Problem:**
```
PNC: RSI 72.3 outside range [30, 70]
```

**What it means:** Stock is overbought (RSI > 70)

**Fix:**

```python
from config import screening_config
screening_config.max_rsi = 80  # Was 70, now more lenient
```

---

## Progressive Relaxation Strategy

Try this sequence to find the right balance:

### **Step 1: Relax Technical Filters First**

```python
from config import entry_config

# Widen acceptable ranges
entry_config.z_score_max = 1.0  # Allow slightly overbought
entry_config.require_positive_momentum = False  # Accept any trend
entry_config.preferred_entry_days = list(range(3, 15))  # 3-14 days before ex-div
```

**Re-run backtest → Check if trades generated**

---

### **Step 2: If Still No Trades - Disable Filters**

```python
# Minimal filtering (for testing only!)
entry_config.use_z_score_filter = False
entry_config.require_positive_momentum = False
entry_config.use_volatility_filter = False
```

**Re-run → Should generate trades now**

---

### **Step 3: Add Filters Back Gradually**

Once you see trades with no filters:

```python
# Start with just RSI
entry_config.use_z_score_filter = False
entry_config.require_positive_momentum = False
entry_config.use_volatility_filter = False
# (RSI is always on via screening_config)
```

Run backtest → Record Sharpe ratio

```python
# Add momentum
entry_config.require_positive_momentum = True
```

Run backtest → Did Sharpe improve?

Keep adding filters one-by-one, testing performance each time.

---

## Confidence-Based Position Sizing (Advanced)

Instead of pass/fail filters, scale position size by confidence:

```python
from config import risk_config

# In config.py, you can modify max_position_size based on quality_score
# Higher quality = larger position

def calculate_position_size_with_confidence(base_size, quality_score):
    # quality_score is 0-100
    # Scale position: 50% size at quality=50, 100% at quality=70, 150% at quality=90+
    confidence_multiplier = (quality_score - 50) / 40  # 0.0 to 1.0+ range
    return base_size * max(0.5, min(1.5, confidence_multiplier))
```

---

## Recommended Starting Config (For Testing)

```python
from config import entry_config, screening_config, use_relaxed_screening

# Use relaxed screening
use_relaxed_screening()

# Relaxed technical filters
entry_config.z_score_min = -3.0
entry_config.z_score_max = 2.0
entry_config.require_positive_momentum = False
entry_config.preferred_entry_days = list(range(2, 12))  # 2-11 days
entry_config.max_realized_vol = 0.50  # 50% annualized (was 30%)

# Run backtest
from backtester import Backtester
bt = Backtester(start_date='2024-08-01', end_date='2024-10-31', initial_capital=100_000)
results = bt.run_backtest(mode='full')
```

**Expected:** Should generate trades now!

---

## Interpreting Results

**Good signs:**
- ✅ 10+ trades over 3 months
- ✅ Sharpe ratio > 0.5
- ✅ Win rate > 50%
- ✅ Max drawdown < 15%

**Red flags:**
- ❌ <5 trades (filters too strict)
- ❌ Sharpe < 0 (losing money)
- ❌ Win rate < 40% (bad signals)
- ❌ Max drawdown > 30% (too risky)

---

## Next Steps

1. **Start relaxed** → Get trades flowing
2. **Add filters gradually** → Test each one's impact on Sharpe
3. **Optimize thresholds** → Find sweet spot between trade frequency and quality
4. **Test longer periods** → Verify robustness (6-12 months)

Once you find settings that work, save them and test on different time periods to verify they're not overfit!
