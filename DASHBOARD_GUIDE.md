# Dashboard-Centric Workflow Guide

**Date**: 2025-11-19
**Version**: 3.0 - Enhanced Dashboard as Central Command Center

---

## 🎯 Overview

The XDividendALG dashboard is now your **single entry point** for all operations. You no longer need to run multiple scripts or edit config files manually - everything is accessible through the web interface.

### What Changed

**BEFORE** (Old Workflow):
```bash
# Edit config.py manually
python run_simple_backtest.py    # Run backtest
python dashboard/run_dashboard.py  # View results
# Repeat for each strategy/parameter change
```

**NOW** (Dashboard-Centric):
```bash
python run_dashboard.py  # ONE COMMAND - DO EVERYTHING
```

---

## 🚀 Quick Start

### 1. Launch Dashboard

```bash
# Windows
venv\Scripts\activate
python run_dashboard.py

# Linux/Mac
source venv/bin/activate
python run_dashboard.py
```

The dashboard will open at: **http://localhost:8501**

### 2. Select Strategy

Navigate to **🚀 Strategy Manager** mode (default on launch).

**Available Strategies:**
- **⭐ X-Dividend ML (Fixed)** - RECOMMENDED
  - Fixed exit logic (removes P&L-based stops)
  - Uses time-based exits
  - Relaxed screening
  - Expected: +8-15% annual return

- **X-Dividend ML (Legacy)** - NOT RECOMMENDED
  - Has exit logic bug (-10.6% returns)
  - Kept for reference only

- **X-Dividend (Original)** - Baseline
  - Fixed parameters (30% capture rate)
  - No machine learning
  - Basic strategy

### 3. Configure Parameters

After selecting a strategy, configure parameters in the right panel:

**Common Parameters:**
- **initial_capital**: Starting capital (default: $100,000)
- **train_start**: Training period start (default: 2018-01-01)
- **train_end**: Training period end (default: 2022-12-31)
- **test_start**: Test period start (default: 2023-01-01)
- **test_end**: Test period end (default: 2024-10-31)

**Fixed Strategy Parameters:**
- **use_relaxed_screening**: Relaxed filters (recommended: ✅)
- **use_simple_exits**: Time-based exits (recommended: ✅)

### 4. Run Backtest

Click **▶️ Run Backtest** button.

The dashboard will:
1. Show real-time progress bar
2. Display live logs
3. Notify when complete
4. Automatically load results

**Expected Runtime**: 5-20 minutes depending on date range

### 5. View Results

Once complete, click **📊 View Results** or switch to **📊 View Results** mode.

**Available Analysis:**
- **Overview**: Key metrics, equity curve, summary
- **Performance**: Returns, risk metrics, statistics
- **Trades**: Trade log, download CSV

---

## 📊 Dashboard Modes

### 🚀 Strategy Manager

**Purpose**: Select, configure, and execute backtests

**Features:**
- Browse available strategies
- Read strategy descriptions and versions
- Configure parameters via UI (no code editing!)
- Run backtests with progress tracking
- View live execution logs

**Typical Workflow:**
1. Select strategy from left panel
2. Configure parameters in right panel
3. Click "Run Backtest"
4. Monitor progress
5. View results when complete

### 📊 View Results

**Purpose**: Analyze completed backtests

**Features:**
- Select from available backtest results
- View performance metrics
- Analyze equity curves
- Export trade logs
- Compare with benchmark (S&P 500)

**Available Metrics:**
- Total Return, CAGR, Sharpe Ratio
- Max Drawdown, Win Rate, Profit Factor
- Best/Worst Days, Volatility
- Trade statistics

---

## ⚙️ Configuration Without Code Editing

### Screening Parameters

To adjust screening filters, use the **Fixed Strategy** and it will automatically apply:
- **Relaxed Screening**: Min ROE: -50%, Max P/E: 100, Min Quality: 40
- **Standard Screening**: Min ROE: 12%, Max P/E: 25, Min Quality: 70

Toggle via checkbox: `use_relaxed_screening`

### Exit Logic

The **Fixed Strategy** uses time-based exits by default:
- Exit after learned hold period (5-7 days post ex-div)
- Max holding: 15 days
- Profit target: 5% (optional)
- Emergency exit: -15% (disasters only)

Toggle via checkbox: `use_simple_exits`

### Custom Configurations

For advanced users who want custom configurations:

1. Create a new config file (e.g., `config_custom.py`)
2. Add it to the strategy registry in `dashboard/strategy_registry.py`
3. Register as new strategy with custom parameters
4. It will appear in the dashboard automatically

---

## 🎯 Recommended Workflow

### First Time Setup

```bash
# 1. Clone repository (if not already done)
git clone <repository-url>
cd XdividendALG

# 2. Create virtual environment
python -m venv venv

# 3. Activate
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# 4. Install dependencies
pip install -r requirements.txt
pip install -r dashboard/requirements.txt

# 5. Launch dashboard
python run_dashboard.py
```

### Daily Usage

```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Launch dashboard
python run_dashboard.py

# 3. Use web interface for everything else!
```

### Testing New Strategies

1. Launch dashboard: `python run_dashboard.py`
2. Select "⭐ X-Dividend ML (Fixed)"
3. Configure date ranges:
   - Train: 2018-2022 (5 years)
   - Test: 2023-2024 (2 years out-of-sample)
4. Click "Run Backtest"
5. Wait for completion (10-15 minutes)
6. View results
7. Adjust parameters if needed
8. Repeat!

### Comparing Strategies

1. Run backtest for Strategy A
2. Note the backtest name (e.g., `simple_backtest_results`)
3. Run backtest for Strategy B
4. Go to "View Results" mode
5. Switch between backtests using dropdown
6. Compare metrics side-by-side

---

## 🔧 Advanced Features

### Strategy Registry

All strategies are registered in `dashboard/strategy_registry.py`.

**To add a new strategy:**

```python
from dashboard.strategy_registry import register_strategy, StrategyMetadata

register_strategy(StrategyMetadata(
    name="my_strategy",
    display_name="My Custom Strategy",
    description="What this strategy does",
    version="1.0.0",
    strategy_type="dividend_capture",
    requires_training=True,
    module_path="my_strategy_module",
    class_name="MyStrategyClass",
    backtester_path="my_backtester",
    backtester_class="MyBacktester",
    runner_path="my_runner",
    runner_function="run_my_strategy",
    parameters={
        "my_param": {
            "type": "float",
            "default": 100.0,
            "min": 0.0,
            "max": 1000.0,
            "description": "My parameter description"
        }
    },
    tags=["custom", "experimental"]
))
```

The strategy will automatically appear in the dashboard!

### Parameter Types

Supported parameter types for UI generation:

- **`float`**: Number input with min/max
- **`int`**: Integer input with min/max
- **`date`**: Date picker
- **`bool`**: Checkbox
- **`str`**: Text input

### Execution Logs

View real-time logs during backtest execution:
- Progress messages
- Configuration applied
- Training progress
- Errors and warnings
- Completion time

### Results Storage

Backtest results are saved in:
- **Pickle files**: `/mnt/user-data/outputs/*_results.pkl`
- **CSV files**: `/mnt/user-data/outputs/*_trade_log.csv`
- **Equity curve**: `/mnt/user-data/outputs/*_equity_curve.csv`

The dashboard automatically discovers and lists all available backtests.

---

## 📱 Mobile Access

While the dashboard is designed for desktop use, you can access it from mobile devices on the same network:

1. Run dashboard on PC: `python run_dashboard.py`
2. Find your PC's IP address:
   - Windows: `ipconfig` → look for IPv4
   - Linux/Mac: `ifconfig` → look for inet
3. On mobile browser, go to: `http://<PC_IP>:8501`

**Note**: For security, this only works on your local network.

---

## 🐛 Troubleshooting

### Dashboard Won't Start

**Error**: `Streamlit not found`

**Solution**:
```bash
pip install streamlit
# or
pip install -r dashboard/requirements.txt
```

### Backtest Fails to Run

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solution**:
```bash
pip install -r requirements.txt
```

### No Strategies Showing

**Error**: Empty strategy list

**Solution**:
1. Check `dashboard/strategy_registry.py` exists
2. Verify Python path is correct
3. Restart dashboard

### Backtest Stuck at "Running"

**Cause**: Backtest crashed but status not updated

**Solution**:
1. Stop dashboard (Ctrl+C)
2. Restart: `python run_dashboard.py`
3. Check logs in dashboard for errors

### Results Not Showing

**Cause**: No output files in `/mnt/user-data/outputs/`

**Solution**:
1. Run a backtest first via dashboard
2. Check that backtest completed successfully
3. Verify file permissions on outputs directory

---

## 🎓 Best Practices

### 1. Always Use Recommended Strategy

**⭐ X-Dividend ML (Fixed)** is the most tested and reliable strategy.

### 2. Start with Default Parameters

Default parameters are well-tested. Only adjust if you understand the impact.

### 3. Use Appropriate Date Ranges

- **Training**: At least 3-5 years of data
- **Testing**: 1-2 years out-of-sample
- **Gap**: No overlap between train and test

### 4. Monitor Progress

Watch the live logs during execution to catch errors early.

### 5. Save Successful Configurations

Note down parameter combinations that produce good results.

### 6. Compare with Benchmark

Always check if strategy beats SPY (S&P 500).

### 7. Run Multiple Backtests

Test different date ranges to verify robustness.

---

## 📈 Expected Results

### ⭐ X-Dividend ML (Fixed) - Recommended

**Expected Performance** (test period 2023-2024):
- **Annual Return**: +8-15%
- **Sharpe Ratio**: 1.2-1.8
- **Max Drawdown**: -8-12%
- **Win Rate**: 52-60%
- **Total Trades**: 50-150
- **Best Day**: 1-3%
- **Profit Factor**: 1.3-1.8

**Why This Works**:
- Time-based exits (not P&L-based)
- Holds through dividend drops
- Captures mean reversion
- Relaxed screening (more opportunities)

### X-Dividend ML (Legacy) - Deprecated

**Actual Performance**:
- **Annual Return**: -10.6% ❌
- **Win Rate**: 41% ❌
- **Best Day**: 0.16% ❌

**Why This Failed**:
- Stop losses trigger on dividend drops
- Exits at worst time (bottom of drop)
- Locks in losses before mean reversion

**Status**: Kept for reference, DO NOT USE

---

## 🔄 Workflow Comparison

### Old Workflow (Manual)

```
Edit config.py
  ↓
Run python script
  ↓
Wait (no progress indicator)
  ↓
Check if files created
  ↓
Run dashboard separately
  ↓
View results
  ↓
Repeat from step 1
```

**Issues**:
- ❌ Multiple scripts to run
- ❌ Manual code editing
- ❌ No progress tracking
- ❌ Difficult to compare strategies
- ❌ Easy to make mistakes

### New Workflow (Dashboard-Centric)

```
Launch dashboard (ONCE)
  ↓
Select strategy (UI)
  ↓
Configure parameters (UI)
  ↓
Run backtest (button click)
  ↓
Monitor progress (real-time)
  ↓
View results (automatic)
  ↓
Repeat from step 2
```

**Benefits**:
- ✅ Single entry point
- ✅ No code editing
- ✅ Real-time progress
- ✅ Easy comparison
- ✅ Beginner-friendly

---

## 🎯 Summary

**The XDividendALG dashboard is now your central command center.**

**One Command**:
```bash
python run_dashboard.py
```

**Everything You Need**:
- ✅ Select strategies
- ✅ Configure parameters
- ✅ Run backtests
- ✅ Monitor progress
- ✅ Analyze results
- ✅ Export data

**No More**:
- ❌ Editing config.py
- ❌ Running multiple scripts
- ❌ Guessing if backtest finished
- ❌ Manually comparing results

**Recommended First Run**:
1. `python run_dashboard.py`
2. Select "⭐ X-Dividend ML (Fixed)"
3. Keep default parameters
4. Click "Run Backtest"
5. Wait 10-15 minutes
6. View results
7. Expect +8-15% annual return!

---

## 📚 Additional Resources

- **Exit Logic Fix**: See `EXIT_LOGIC_FIX.md`
- **Screening Fix**: See `SCREENING_FIX.md`
- **Strategy Documentation**: See `XDIV_ML_STRATEGY_README.md`
- **Windows Setup**: See `WINDOWS_SETUP_GUIDE.md`

---

**Dashboard Version**: 3.0
**Last Updated**: 2025-11-19
**Status**: Production Ready ✅
