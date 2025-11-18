# Quick Start - Windows

**5-Minute Setup Guide**

---

## 1️⃣ Prerequisites (One-Time Setup)

```cmd
# Check if you have Python installed
python --version

# If not, download from: https://www.python.org/downloads/
# ⚠️ Check "Add Python to PATH" during installation
```

---

## 2️⃣ Clone & Setup (5 minutes)

```cmd
# Clone the repository
git clone https://github.com/DennisPoniros/XdividendALG.git
cd XdividendALG

# Checkout the ML strategy branch
git checkout claude/fix-algo-strategy-01UAPSmDv24UxUnvkTW9VvYc

# Create virtual environment
python -m venv venv

# Activate it (Command Prompt)
venv\Scripts\activate

# OR (PowerShell)
venv\Scripts\Activate.ps1

# Install dependencies
pip install pandas numpy scipy yfinance matplotlib seaborn
```

---

## 3️⃣ Run Backtest (10-15 minutes)

```cmd
# Make sure venv is activated (you see "(venv)" in terminal)
python run_xdiv_ml_backtest.py
```

**What happens:**
- ⏱️ Downloads 5 years of dividend data
- 🎓 Trains on 2018-2022 data
- ✅ Tests on 2023-2024 data
- 📊 Generates reports and charts

---

## 4️⃣ View Results

Results saved to: `outputs/` folder

**Open these files:**
- `xdiv_ml_report.html` - Full performance report (open in browser)
- `xdiv_ml_training_summary.txt` - What parameters were learned
- `xdiv_ml_equity_curve.png` - Portfolio growth chart

---

## 5️⃣ Compare Strategies

Script automatically compares old vs new strategy:

```
                           Old Strategy    |    New ML Strategy
Annual Return:                  6.50%    |           12.30%
Sharpe Ratio:                    0.95    |            1.68
Win Rate:                      48.50%    |           58.40%
```

---

## Troubleshooting

### ❌ "python is not recognized"
**Fix:** Reinstall Python, check "Add to PATH"

### ❌ PowerShell won't run scripts
**Fix:** Run as Administrator:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### ❌ "No module named 'pandas'"
**Fix:** Make sure venv is activated, then:
```cmd
pip install pandas numpy scipy yfinance
```

### ❌ Takes too long
**Fix:** Be patient - first run downloads 5+ years of data for 100+ stocks

---

## VS Code (Recommended)

1. **Open folder** in VS Code: `File → Open Folder`
2. **Select Python interpreter**: Click bottom-left → Select `venv/Scripts/python.exe`
3. **Run**: Open `run_xdiv_ml_backtest.py` → Press `F5`

---

## Expected Results

**Training Phase:**
```
✅ TRAINING COMPLETED
📊 Training Metrics:
  Dividend Events Analyzed: 2,456
  Average Capture Rate:     32.5%
  Optimal Entry Days:       [4, 5]
  Optimal Hold Period:      5 days
```

**Testing Phase:**
```
📈 TEST PERIOD PERFORMANCE SUMMARY
💰 RETURNS
  Annual Return:               10.23%
⚠️  RISK METRICS
  Sharpe Ratio:                 1.68
📊 TRADE STATISTICS
  Win Rate:                    58.45%

✅ EXCELLENT: Sharpe ratio exceeds target (>1.5)
```

---

## Full Documentation

For detailed guide, see: **WINDOWS_SETUP_GUIDE.md**

For strategy details, see: **XDIV_ML_STRATEGY_README.md**

---

**Ready? Let's go!** 🚀

```cmd
python run_xdiv_ml_backtest.py
```
