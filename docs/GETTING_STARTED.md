# 🚀 Quick Start Guide - Dividend Capture Algorithm

## ⚡ 3-Minute Setup

### Step 1: Get Alpaca API Keys (Optional but Recommended)

1. Go to https://alpaca.markets
2. Sign up for free account
3. Enable paper trading
4. Generate API keys (View → API Keys)
5. Copy Key ID and Secret Key

### Step 2: Configure API Keys

Edit `config.py` line 16-19:

```python
ALPACA_CONFIG = {
    'API_KEY': 'PKxxxxxxxxx',        # Paste your key here
    'SECRET_KEY': 'xxxxxxxxxx',      # Paste your secret here
    'BASE_URL': 'https://paper-api.alpaca.markets'
}
```

**Skip this if you don't have keys** - the system will work with yfinance only.

### Step 3: Install Dependencies

```bash
pip install pandas numpy scipy yfinance matplotlib seaborn scikit-learn statsmodels alpaca-trade-api
```

Or use the requirements file:
```bash
pip install -r requirements.txt
```

### Step 4: Run Your First Backtest

```bash
python main.py
```

When prompted, choose option **2** (Train/Test Split - recommended)

### Step 5: View Results

Check `/mnt/user-data/outputs/` for:
- `backtest_report.html` - Open this in a browser!
- `equity_curve.png` - Portfolio growth
- `drawdown.png` - Risk visualization
- `trade_log.csv` - All trades

---

## 📊 What You Should See

### Good Results (Strategy is Working)
✅ Sharpe Ratio > 1.5  
✅ Annual Return > 10%  
✅ Win Rate > 55%  
✅ Max Drawdown < 15%  

### Needs Tuning
⚠️ Sharpe Ratio < 1.0  
⚠️ Win Rate < 50%  
⚠️ Max Drawdown > 20%  

If you see warning signs, try:
1. Run parameter sweep: `python main.py sweep`
2. Adjust parameters in `config.py`
3. Re-run backtest

---

## 🎯 Parameter Tuning Quick Reference

**Want higher returns?** (but more risk)
- Increase `profit_target_multiple` to 2.0
- Increase `max_position_pct` to 0.03

**Want lower risk?** (but lower returns)
- Decrease `max_position_pct` to 0.01
- Increase `min_quality_score` to 80
- Decrease `max_positions` to 15

**Want more trades?**
- Lower `min_quality_score` to 65
- Expand `preferred_entry_days` to [2,3,4,5,6]

**Want fewer, higher quality trades?**
- Increase `min_quality_score` to 80
- Narrow `preferred_entry_days` to [4,5]

---

## 🔧 Common Issues & Fixes

### "No dividend events found"
**Fix:** Expand date range in config.py:
```python
train_start: str = '2020-01-01'  # Start earlier
test_end: str = '2024-12-31'     # End later
```

### "Alpaca connection failed"
**Fix:** Either:
1. Check API keys in config.py
2. Or just use yfinance (comment out Alpaca config)

### "Not enough trades"
**Fix:** Lower screening threshold:
```python
min_quality_score: float = 60.0  # Was 70
```

### "Performance is poor"
**Fix:** Run parameter optimization:
```bash
python main.py sweep
```

---

## 📈 Next Steps After First Backtest

### If Results are Good (Sharpe > 1.5)
1. ✅ Run walk-forward validation: `python main.py` → option 3
2. ✅ Test different time periods
3. ✅ Move to paper trading

### If Results Need Work
1. 🔧 Run parameter sweep: `python main.py sweep`
2. 🔧 Adjust based on sweep results
3. 🔧 Re-run backtest
4. 🔧 Repeat until Sharpe > 1.5

### Ready for Paper Trading?
1. Ensure backtest Sharpe > 1.5 consistently
2. Connect Alpaca paper trading account
3. Run live for 60+ days
4. Monitor daily performance
5. Compare to backtest results

---

## 📚 Understanding Your Results

### Sharpe Ratio (Risk-Adjusted Return)
- **> 2.0:** Excellent
- **1.5-2.0:** Very Good (target)
- **1.0-1.5:** Good
- **< 1.0:** Needs improvement

### Win Rate
- **> 60%:** Excellent
- **55-60%:** Very Good (target)
- **50-55%:** Good
- **< 50%:** Needs improvement

### Maximum Drawdown
- **< 10%:** Excellent
- **10-15%:** Good (target)
- **15-20%:** Acceptable
- **> 20%:** Too risky

### Annual Return
- **> 15%:** Excellent
- **10-15%:** Very Good (target)
- **5-10%:** Good
- **< 5%:** Below expectations

---

## 💡 Pro Tips

1. **Always start with Train/Test Split** (option 2) - detects overfitting
2. **Run parameter sweep** before committing to settings
3. **Paper trade for 60+ days** before going live
4. **Start small** - use 25% of intended capital initially
5. **Monitor daily** - don't set and forget
6. **Review monthly** - markets change, adapt parameters

---

## 🆘 Need Help?

1. Check `README.md` for detailed documentation
2. Review `PROJECT_SUMMARY.md` for system overview
3. Read troubleshooting section in README
4. Check data availability: `python data_manager.py`
5. Run validation: `python main.py validate`

---

## ✅ Pre-Flight Checklist

Before running backtest:
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] API keys configured (optional but recommended)
- [ ] Output directory exists (`/mnt/user-data/outputs/`)
- [ ] Config reviewed (`config.py`)
- [ ] Date ranges set correctly

Before paper trading:
- [ ] Backtest Sharpe > 1.5
- [ ] Win rate > 55%
- [ ] Tested multiple time periods
- [ ] Walk-forward validation passed
- [ ] Understand all parameters
- [ ] Have monitoring plan

Before live trading:
- [ ] Paper trading 60+ days successful
- [ ] Performance matches backtest (±20%)
- [ ] Risk controls tested
- [ ] Emergency stop plan in place
- [ ] Starting with <25% of capital
- [ ] Daily monitoring system ready

---

## 🎯 Your First Command

```bash
cd dividend_algo
python main.py
```

**Choose option 2 when prompted**

Then open: `/mnt/user-data/outputs/backtest_report.html`

---

**Good luck!** 📈🚀

*Remember: Past performance doesn't guarantee future results. Trade responsibly.*
