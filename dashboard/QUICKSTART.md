# Dashboard Quick Start Guide

## Installation

```bash
# Install required packages
pip install -r dashboard/requirements.txt

# Optional: Install yfinance for S&P 500 benchmark comparison
pip install yfinance
```

## Launch Dashboard

**Simplest method:**
```bash
python dashboard/run_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

**Custom port:**
```bash
python dashboard/run_dashboard.py --port 8080
```

**Remote access:**
```bash
python dashboard/run_dashboard.py --host 0.0.0.0 --port 8080
```

## First Time Usage

### 1. Run a Backtest

Before using the dashboard, you need to run at least one backtest:

```bash
# From the project root
python main.py
```

Select a backtest mode when prompted. The results will be saved to `/mnt/user-data/outputs/`

### 2. View Results in Dashboard

Once you have backtest results:
1. Launch the dashboard: `python dashboard/run_dashboard.py`
2. The dashboard will automatically load the most recent backtest
3. Use the sidebar to select different backtests

## Dashboard Features

### 📊 Overview Tab
- **Key Metrics**: Total Return, Sharpe Ratio, Max Drawdown, Win Rate, Total Trades
- **Equity Curve**: Interactive chart with entry/exit markers
- **S&P 500 Comparison**: Benchmark overlay (if yfinance installed)
- **Drawdown Chart**: Underwater plot showing drawdown periods
- **Risk-Adjusted Metrics**: Sharpe, Sortino, Omega, Calmar ratios

### 📈 Performance Tab
- **Rolling Metrics**: 30-day rolling Sharpe and volatility
- **Returns Distribution**: Histogram with normal curve overlay
- **Win Rate Analysis**: Trades grouped by P&L buckets
- **Performance Heatmap**: Trade performance by day/month
- **Distribution Statistics**: Skewness, kurtosis, VaR, CVaR

### 💼 Trades Tab
- **Trade Summary**: Total, winning, losing trades
- **Outlier Analysis**: Top 10 best and worst trades
- **Complete Trade Log**: Sortable, filterable table
- **CSV Export**: Download complete trade history

### 🎯 Risk Tab
- **Risk Metrics**: VaR, CVaR, drawdown duration
- **Leverage Charts**: Leverage ratio and capital utilization
- **Underwater Time**: Time spent in drawdown
- **Drawdown Details**: Duration, depth, recovery

### ⚙️ Configuration Tab
Coming soon - adjust strategy parameters from the interface

## Metrics Reference

### What's Good?
- **Sharpe Ratio**: > 1.5 is good, > 2.0 is excellent
- **Win Rate**: > 55% is good
- **Profit Factor**: > 1.5 is good
- **Max Drawdown**: < -15% is acceptable
- **Omega Ratio**: > 1.0 is profitable

### Color Coding
- 🟢 **Green**: Positive/profitable
- 🔴 **Red**: Negative/losses
- 🔵 **Blue**: Neutral/strategy
- 🟠 **Orange**: Benchmark

## Tips

### Performance
- Enable auto-refresh only when needed (sidebar control)
- For large backtests (>10k trades), refresh interval should be higher
- Close unused browser tabs

### Analysis
- Hover over charts for detailed tooltips
- Use Plotly controls to zoom, pan, and export charts
- Right-click charts to download as PNG

### Comparison
- Use the backtest selector in sidebar to compare different runs
- Note the modification date to find recent backtests
- Trade log CSV export allows external analysis

## Troubleshooting

### Dashboard won't start
```bash
# Ensure Streamlit is installed
pip install streamlit

# Check version
streamlit --version
```

### No data showing
- Ensure you've run at least one backtest
- Check `/mnt/user-data/outputs/` for result files
- Select different backtest in sidebar

### Slow performance
- Reduce auto-refresh frequency
- Close other browser tabs
- Restart dashboard

### Benchmark not loading
- Install yfinance: `pip install yfinance`
- Check internet connection
- Dashboard works fine without benchmark data

## Next Steps

1. ✅ **Run your first backtest**: `python main.py`
2. ✅ **Launch dashboard**: `python dashboard/run_dashboard.py`
3. ✅ **Analyze results**: Explore all tabs
4. ✅ **Optimize strategy**: Use insights to improve parameters
5. ✅ **Compare runs**: Test different configurations

## Advanced Usage

### Live Mode (Coming Soon)
Monitor backtest execution in real-time:
```python
from dashboard.data_interface import LiveBacktestStreamer

# In your backtest code
streamer = LiveBacktestStreamer(data_interface)
streamer.start()

# Update during backtest loop
streamer.update(date, equity, trades, positions, position_value)
```

### Custom Metrics
Add your own metrics in `dashboard/metrics.py`:
```python
def calculate_custom_metric(self, returns):
    return your_calculation
```

### Custom Visualizations
Add new charts in `dashboard/visualizations.py`:
```python
def plot_custom_chart(self, data):
    fig = go.Figure()
    # Your plotly code
    return fig
```

## Support

- 📖 Full documentation: `dashboard/README.md`
- 🧪 Run tests: `python dashboard/test_dashboard.py`
- 🔍 Check logs for errors
- 📝 Review example notebooks

---

**Dashboard Version**: 0.1.0
**Last Updated**: 2024
