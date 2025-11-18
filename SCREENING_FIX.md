# SCREENING CONFIGURATION FIX

## Problem Identified

**Only 2 trades in 11 months** - screening filters are rejecting 99.9% of candidates.

---

## Root Cause

Original `screening_config` has **unrealistic requirements**:

| Filter | Original | Why Too Strict |
|--------|----------|----------------|
| **ROE** | ≥ 12% | Excludes most companies |
| **P/E Ratio** | ≤ 25 | Excludes all growth stocks |
| **Quality Score** | ≥ 70/100 | Very high bar |
| **Debt/Equity** | ≤ 0.50 | Too conservative |
| **Dividend Yield** | 2-8% only | Narrow range |
| **Positive Momentum** | Required | Excludes mean reversion setups |

**Result**: Almost no candidates pass → No trades → Strategy can't work

---

## Solution: RELAXED SCREENING

New default configuration focuses on **logical, minimal filters**:

### Comparison Table

| Filter | STRICT (Original) | **RELAXED (New Default)** | Reasoning |
|--------|-------------------|---------------------------|-----------|
| **Min Market Cap** | $1B | **$500M** | Include mid-caps |
| **Min Volume** | 500k | **100k** | More opportunities |
| **Div Yield Range** | 2-8% | **1-20%** | Include high-yield |
| **Min Div History** | 5 years | **2 years** | Recent payers OK |
| **Days to Ex-Div** | 3-20 | **2-25** | Wider window |
| **Min Quality Score** | 70/100 | **40/100** | Basic quality |
| **Max Payout Ratio** | 80% | **150%** | Allow REITs |
| **Max Debt/Equity** | 0.5 | **2.0** | Normal leverage OK |
| **Min ROE** | 12% | **-50%** | Just not bankrupt |
| **Max P/E** | 25 | **100** | Allow growth stocks |
| **Min RSI** | 30 | **20** | More oversold OK |
| **Max RSI** | 70 | **80** | Less restrictive |
| **Positive Momentum** | Required | **Not required** | Allow mean reversion |
| **Max Beta** | 1.2 | **3.0** | Allow volatile stocks |

---

## Philosophy Change

### ❌ OLD APPROACH (Doesn't Work)
```
Heavy Fundamental Screening
    ↓
Few candidates (1-5 per month)
    ↓
Strategy filters can't work
    ↓
Almost no trades
```

### ✅ NEW APPROACH (Correct)
```
Minimal Fundamental Screening (liquidity + dividend exists)
    ↓
Many candidates (20-50 per week)
    ↓
Strategy Filters Do Selection (RSI, Z-score, momentum, learned params)
    ↓
Quality trades (5-15 per month)
```

---

## What Gets Through Now

### STRICT (Original)
- Perfect companies only
- Low debt, high ROE, reasonable P/E
- **Result**: Almost nothing passes

### RELAXED (New)
- Has dividend (any yield)
- Has liquidity (can trade it)
- Basic financial health (not bankrupt)
- **Strategy decides** based on:
  - RSI (oversold?)
  - Z-score (mean reversion?)
  - Momentum
  - Learned capture rates
  - Expected return calculation

---

## Expected Impact

### Before (Strict)
```
11 months = 2 trades
Annual: ~2-3 trades
Impossible to validate strategy
```

### After (Relaxed)
```
Expected: 50-150 trades/year
~4-12 trades/month
Sufficient for statistical validation
Strategy can actually work
```

---

## Implementation

### Automatic (No Action Needed)

The fix is **already applied** in `run_xdiv_ml_backtest.py`:

```python
from config_relaxed import use_relaxed_screening

def run_xdiv_ml_backtest():
    # Automatically uses relaxed screening
    use_relaxed_screening()
    # ... rest of backtest
```

### Manual Control (If Needed)

**Option 1: Use Relaxed (Default)**
```python
from config_relaxed import use_relaxed_screening
use_relaxed_screening()  # Already done automatically
```

**Option 2: Use Minimal (Maximum Opportunities)**
```python
from config_relaxed import use_minimal_screening
use_minimal_screening()  # Even looser
```

**Option 3: Use Original Strict (Not Recommended)**
```python
# Don't call use_relaxed_screening()
# Will get original strict filters (only 2 trades)
```

---

## Verification

### Check Current Settings

```python
from config import screening_config

print(f"Min ROE: {screening_config.min_roe*100}%")
print(f"Max P/E: {screening_config.max_pe_ratio}")
print(f"Min Quality: {screening_config.min_quality_score}")
```

**Before fix**:
```
Min ROE: 12%
Max P/E: 25
Min Quality: 70
```

**After fix**:
```
Min ROE: -50%
Max P/E: 100
Min Quality: 40
```

---

## Key Insight

**The strategy's edge comes from**:
1. ✅ Dividend capture mechanics (learned rates)
2. ✅ Mean reversion signals (RSI, Z-score)
3. ✅ Technical entry filters
4. ✅ Learned optimal parameters

**NOT from**:
- ❌ Finding "perfect" companies
- ❌ Strict fundamental filters
- ❌ Only buying low P/E stocks

---

## What Strategy Still Filters

Even with relaxed screening, the **strategy's entry filters** still apply:

### Entry Filters (Where Real Selection Happens)
```python
1. RSI in learned optimal range (e.g., 30-65)
2. Z-score below learned threshold (e.g., -1.25)
3. Days to ex-div in learned window (e.g., [4, 5])
4. Expected return > learned minimum (e.g., 1.35%)
5. Volatility < 30%
6. Optional: Positive momentum (now optional)
```

### Risk Management
```python
1. Position sizing via Kelly Criterion
2. Stop losses (2% hard stop or 1x dividend)
3. Max holding period (10 days)
4. Learned optimal hold period (e.g., 5-7 days)
```

**So we still get quality trades**, just from a larger opportunity set.

---

## Files Changed

1. ✅ `config_relaxed.py` - New relaxed configs
2. ✅ `run_xdiv_ml_backtest.py` - Auto-applies relaxed screening
3. ✅ `SCREENING_FIX.md` - This document

---

## Next Steps

### For Your Running Backtest

**Kill it and restart** with new screening:

```cmd
# Stop current backtest (Ctrl+C)

# Pull latest changes
git pull origin claude/fix-algo-strategy-01UAPSmDv24UxUnvkTW9VvYc

# Run again (will use relaxed screening)
python run_xdiv_ml_backtest.py
```

### Expected Output

You should now see:
```
✅ Switched to RELAXED screening
   Min ROE: -50% (was 12%)
   Max P/E: 100 (was 25)
   Min Quality: 40/100 (was 70)
   Min Div Yield: 1.0% (was 2%)

🔍 Screening 156 dividend candidates...
✅ 47 stocks passed screening
🎯 Generating entry signals for 47 candidates...
✅ Generated 12 entry signals
```

**Instead of**:
```
🔍 Screening 156 dividend candidates...
⚠️  No stocks passed screening criteria
✅ Generated 0 entry signals
```

---

## Summary

**Problem**: Only 2 trades because screening too strict

**Fix**: Relaxed screening (automatically applied)

**Philosophy**: Let strategy filters do selection, not fundamental screening

**Expected**: 50-150 trades/year (sufficient for validation)

**Action**: Pull latest code and re-run backtest

---

**This fix is critical** - the strategy literally cannot work without sufficient trading opportunities!
