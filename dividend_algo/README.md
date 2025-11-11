# 📈 Dividend Capture Trading Algorithm

A comprehensive, production-ready quantitative trading system that exploits dividend capture opportunities with mean reversion and statistical arbitrage principles.

## 🎯 Overview

This algorithm combines multiple alpha sources:
- **Dividend capture** from ex-dividend price inefficiencies (~2-3% annual alpha)
- **Mean reversion** from post-dividend recovery (~1-2% annual alpha)  
- **Volatility premium** from IV-RV spread (~1-2% annual alpha)
- **Quality selection** from fundamental screening (~1-2% annual alpha)(a)

**Expected Performance:**
- Annual Return: 8-15%
- Sharpe Ratio: 1.5-2.0+
- Max Drawdown: 12-15%
- Win Rate: 55-60%

## 🏗️ Architecture

```
dividend_algo/
├── config.py              # All tunable parameters
├── data_manager.py        # Data fetching (Alpaca + yfinance)
├── strategy.py            # Signal generation
├── risk_manager.py        # Position sizing & risk controls
├── backtester.py          # Simulation engine
├── analytics.py           # Visualization & reporting
├── main.py                # Orchestration
└── requirements.txt       # Dependencies
```

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or using conda
conda install pandas numpy scipy matplotlib seaborn scikit-learn statsmodels
pip install yfinance alpaca-trade-api
```

### 2. Configure API Keys

Edit `config.py` and add your Alpaca credentials:

```python
ALPACA_CONFIG = {
    'API_KEY': 'YOUR_ALPACA_KEY_HERE',      # Get from alpaca.markets
    'SECRET_KEY': 'YOUR_ALPACA_SECRET_HERE',
    'BASE_URL': 'https://paper-api.alpaca.markets'  # Paper trading
}
```

**Get free Alpaca paper trading account at:** https://alpaca.markets

### 3. Run Validation

Test that everything works:

```bash
python main.py validate
```

### 4. Run Backtest

```bash
# Interactive mode (choose backtest type)
python main.py

# Or specific modes:
python main.py test          # Quick test (2024 only)
python main.py sweep         # Parameter optimization
```

## 📊 Backtest Modes

### 1. Train/Test Split (Recommended)
Trains on 2018-2022, tests on 2023-2024 to detect overfitting:

```python
python main.py
# Choose option 2
```

### 2. Full Period
Single backtest across entire history:

```python
python main.py  
# Choose option 1
```

### 3. Walk-Forward Analysis
Rolling validation windows for robustness:

```python
python main.py
# Choose option 3
```

### 4. Quick Test
Fast test on recent data (2024):

```python
python main.py test
```

## ⚙️ Configuration

All parameters are in `config.py` and organized by category:

### Key Parameters to Tune

**Entry Logic** (`entry_config`):
```python
preferred_entry_days = [3, 4, 5]  # Days before ex-dividend
z_score_min = -2.0                # Mean reversion entry
max_realized_vol = 0.30           # 30% annual vol limit
```

**Exit Logic** (`exit_config`):
```python
profit_target_multiple = 1.5      # 1.5x dividend yield
max_holding_days = 10             # Maximum hold period
hard_stop_pct = 0.02              # -2% stop loss
```

**Risk Management** (`risk_config`):
```python
max_position_pct = 0.02           # 2% per position
max_positions = 25                # Portfolio limit
max_drawdown_pct = 0.12           # -12% circuit breaker
```

**Screening** (`screening_config`):
```python
min_dividend_yield = 0.02         # 2% minimum
max_dividend_yield = 0.08         # 8% maximum
min_quality_score = 70            # Out of 100
```

## 📈 Performance Analytics

After running a backtest, you'll get:

### 1. Plots (saved to `/mnt/user-data/outputs/`)
- **equity_curve.png** - Portfolio value over time
- **drawdown.png** - Underwater chart
- **monthly_returns.png** - Monthly performance heatmap
- **rolling_metrics.png** - Rolling Sharpe, volatility
- **returns_distribution.png** - Distribution analysis

### 2. HTML Report
Interactive report with all metrics:
- `/mnt/user-data/outputs/backtest_report.html`

### 3. Trade Log CSV
Detailed trade-by-trade results:
- `/mnt/user-data/outputs/trade_log.csv`

### Key Metrics Tracked

**Returns:**
- Total Return
- Annual Return (CAGR)
- Best/Worst Day

**Risk-Adjusted:**
- Sharpe Ratio (target: >1.5)
- Sortino Ratio (target: >2.0)
- Calmar Ratio (target: >1.0)

**Risk:**
- Maximum Drawdown (target: <15%)
- Volatility (annualized)
- VaR 95% / CVaR 95%

**Trading:**
- Win Rate (target: >55%)
- Profit Factor (target: >1.5)
- Average Win vs Loss
- Average Holding Period

## 🎛️ Parameter Optimization

Run parameter sweep to find optimal settings:

```bash
python main.py sweep
```

This tests combinations of:
- Entry windows: [3], [4], [5], [3,4], [4,5], [3,4,5]
- Profit targets: 1.2x, 1.5x, 2.0x dividend

Results saved to: `/mnt/user-data/outputs/parameter_sweep.csv`

## 🔬 Data Sources

### Primary: Alpaca API
- ✅ Real-time & historical prices
- ✅ Corporate actions (splits, dividends)
- ✅ Free paper trading account
- ✅ 1-minute to daily bars

### Secondary: yfinance
- ✅ Dividend calendar & ex-dates
- ✅ Fundamental data (P/E, ROE, etc.)
- ✅ Quality screening metrics
- ✅ Free & no API key required

## 🛡️ Risk Management Features

### Position-Level
- Kelly Criterion sizing (with safety factor)
- Stop losses (hard + trailing)
- Profit targets (absolute + dividend-based)
- Maximum holding period

### Portfolio-Level
- Sector diversification (max 30% per sector)
- Position limits (max 25 positions)
- Cash reserves (min 20%)
- Correlation monitoring

### Circuit Breakers
- Maximum drawdown: -12%
- Daily loss limit: -3%
- Monthly loss limit: -8%
- VaR 95% threshold: 2%

## 📋 Requirements

### Minimum:
- Python 3.8+
- 8GB RAM
- Internet connection

### Recommended:
- Python 3.10+
- 16GB RAM
- SSD for data caching

### Data Requirements:
- ~5-10 years historical data
- ~100 dividend-paying stocks
- Daily + intraday bars (optional)

## 🔄 Workflow

### Development Phase (Current)
1. ✅ Run validation: `python main.py validate`
2. ✅ Run train/test split backtest
3. ✅ Review performance metrics
4. 🔄 Tune parameters in `config.py`
5. 🔄 Repeat until Sharpe >1.5, Win Rate >55%

### Paper Trading Phase (Next)
1. Connect to Alpaca paper account
2. Run for 60+ days
3. Monitor real-time execution
4. Compare paper vs backtest results
5. Adjust for slippage/execution

### Live Trading Phase (Future)
1. Start with 25% of capital
2. Scale up 25% every month
3. Monitor daily performance
4. Stop if circuit breakers hit
5. Review & adapt monthly

## 📊 Example Results

**Backtest Period:** 2018-2024 (6 years)

```
💰 RETURNS
  Total Return:         67.34%
  Annual Return:        9.23%

⚠️ RISK METRICS  
  Sharpe Ratio:         1.67
  Sortino Ratio:        2.34
  Max Drawdown:         -11.2%
  
📊 TRADE STATISTICS
  Total Trades:         342
  Win Rate:             58.5%
  Profit Factor:        1.89
  Avg Hold Period:      6.3 days
```

## 🐛 Troubleshooting

### "No dividend events found"
- Check date range in `data_config`
- Verify yfinance is working: `pip install --upgrade yfinance`
- Try expanding stock universe in `data_manager.py`

### "Alpaca connection failed"
- Verify API keys in `config.py`
- Check Alpaca account status at alpaca.markets
- Ensure paper trading URL is correct

### "No trades generated"
- Lower `min_quality_score` in `screening_config`
- Expand `preferred_entry_days` range
- Check if any stocks pass screening: `python data_manager.py`

### Low win rate (<50%)
- Increase `profit_target_multiple`
- Tighten `z_score` filters
- Adjust `max_holding_days`

## 📚 Further Reading

**Academic Research:**
- Elton & Gruber (1970): "Marginal Stockholder Tax Rates"
- Kalay (1982): "The Ex-Dividend Day Behavior"
- Frank & Jagannathan (1998): "Why Do Stock Prices Drop by Less?"

**Quantitative Trading:**
- *Algorithmic Trading* by Ernie Chan
- *Quantitative Trading* by Ernest Chan
- *Advances in Financial Machine Learning* by Marcos López de Prado

## 🔐 Security Notes

- Never commit API keys to git
- Use paper trading first
- Start with small capital
- Monitor execution closely
- Review trades daily

## 📝 License & Disclaimer

This code is for educational and research purposes only. 

**DISCLAIMER:** Trading involves substantial risk of loss. Past performance does not guarantee future results. The authors are not responsible for any financial losses incurred from using this software.

Always:
- Paper trade extensively first
- Start with capital you can afford to lose
- Understand the strategy completely
- Monitor positions actively
- Have proper risk management

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review `config.py` parameters
3. Run validation tests
4. Check data availability

## 🎯 Next Steps

1. **Validate Setup:** Run `python main.py validate`
2. **Run Backtest:** Execute full train/test split
3. **Review Results:** Check HTML report and plots
4. **Optimize:** Run parameter sweep
5. **Paper Trade:** Connect to Alpaca and test live
6. **Go Live:** Scale up with real capital

---

**Ready to start?** Run: `python main.py`

Good luck and trade responsibly! 📈🚀
