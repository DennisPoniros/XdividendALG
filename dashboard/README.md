# Backtesting Dashboard

A comprehensive, interactive dashboard for analyzing algorithmic trading strategies in real-time or replay mode.

## Features

### 📊 Real-Time Monitoring
- Live equity curve updates during backtesting
- Real-time trade execution visualization
- Dynamic performance metric calculations
- Current positions and exposure tracking

### 📈 Performance Analytics
- **Risk-Adjusted Metrics**: Sharpe, Sortino, Omega, Calmar ratios
- **Return Analysis**: Total return, CAGR, daily statistics
- **Drawdown Analysis**: Maximum drawdown, duration, recovery
- **Rolling Metrics**: 30-day rolling Sharpe and volatility
- **Distribution Analysis**: Returns histogram with normal curve overlay

### 💼 Trade Analysis
- Complete trade log with filtering and sorting
- Outlier trade identification (best and worst performers)
- Win rate analysis by P&L buckets
- Trade performance heatmap by day/month
- Entry/exit signal visualization on equity curve

### 🎯 Risk Management
- Value at Risk (VaR) and Conditional VaR
- Leverage and capital utilization tracking
- Underwater time analysis
- Drawdown duration statistics

### 🔄 Benchmark Comparison
- S&P 500 comparison (via SPY)
- Alpha and Beta calculation
- Information ratio
- Win rate vs benchmark

### ⚙️ Configuration Interface
- Strategy parameter adjustment (coming soon)
- Date range controls
- Timeframe selection
- Risk parameter tuning

## Installation

1. **Install required packages:**
   ```bash
   pip install -r dashboard/requirements.txt
   ```

2. **Verify installation:**
   ```bash
   python -c "import streamlit; print(streamlit.__version__)"
   ```

## Usage

### Quick Start

**Method 1: Using the launcher script (recommended)**
```bash
python dashboard/run_dashboard.py
```

**Method 2: Direct Streamlit command**
```bash
streamlit run dashboard/app.py
```

**Method 3: Custom host/port**
```bash
python dashboard/run_dashboard.py --host 0.0.0.0 --port 8080
```

### Dashboard Modes

#### 1. Replay Mode (Default)
Analyzes completed backtests from saved results.

```bash
python dashboard/run_dashboard.py
```

- Select from available backtests in the sidebar
- Full historical analysis
- Export capabilities
- No active backtest required

#### 2. Live Mode
Real-time monitoring during backtest execution.

1. Start dashboard in live mode
2. Run backtest with live streaming enabled
3. Watch metrics update in real-time

```python
# In your backtest code
from dashboard.data_interface import BacktestDataInterface, LiveBacktestStreamer

data_interface = BacktestDataInterface(mode='live')
streamer = LiveBacktestStreamer(data_interface)
streamer.start()

# During backtest loop
streamer.update(
    date=current_date,
    equity=current_equity,
    trades=new_trades,
    positions=open_positions,
    position_value=total_position_value
)
```

## Dashboard Tabs

### 📊 Overview
**Main performance summary**
- Key metrics cards (Total Return, Sharpe, Max DD, Win Rate, Trades)
- Equity curve with entry/exit markers
- Benchmark comparison (S&P 500)
- Drawdown chart
- Risk-adjusted performance bar chart
- Summary statistics table

### 📈 Performance
**Detailed performance analysis**
- Rolling 30-day Sharpe ratio and volatility
- Returns distribution histogram
- Win rate analysis by P&L buckets
- Trade performance heatmap (day × month)
- Distribution statistics (skew, kurtosis, VaR)

### 💼 Trades
**Individual trade analysis**
- Trade summary metrics
- Top 10 best and worst trades
- Complete trade log table (sortable, filterable)
- CSV export functionality
- Trade statistics

### 🎯 Risk
**Risk metrics and analysis**
- VaR and CVaR calculations
- Leverage and capital utilization charts
- Drawdown duration analysis
- Underwater time statistics
- Risk exposure tracking

### ⚙️ Configuration
**Strategy settings** (coming soon)
- Parameter adjustment interface
- Date range controls
- Timeframe selection
- Real-time reconfiguration

## Architecture

```
dashboard/
├── __init__.py              # Package initialization
├── app.py                   # Main Streamlit application
├── metrics.py               # Metrics calculation engine
├── visualizations.py        # Plotly chart components
├── data_interface.py        # Data loading and streaming
├── run_dashboard.py         # Launcher script
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

### Components

**app.py**
- Main dashboard application
- Tab management
- User interface
- Session state handling

**metrics.py**
- MetricsCalculator class
- All performance metrics
- Benchmark comparisons
- Statistical calculations

**visualizations.py**
- DashboardVisualizations class
- Plotly chart creation
- Theme management
- Interactive features

**data_interface.py**
- BacktestDataInterface class
- Data loading from files
- Live data streaming
- Caching and optimization

## Metrics Reference

### Return Metrics
- **Total Return**: Overall percentage gain/loss
- **CAGR**: Compound Annual Growth Rate
- **Daily Mean/Std**: Average and volatility of daily returns

### Risk-Adjusted Metrics
- **Sharpe Ratio**: (Return - Risk-free) / Volatility
  - \> 2.0: Excellent
  - \> 1.0: Good
  - < 1.0: Poor

- **Sortino Ratio**: Like Sharpe but uses downside deviation only

- **Omega Ratio**: Probability-weighted gains vs losses
  - \> 1.0: More gains than losses

- **Calmar Ratio**: Return / Max Drawdown

### Drawdown Metrics
- **Max Drawdown**: Largest peak-to-trough decline
- **Drawdown Duration**: Days from peak to recovery
- **Underwater %**: Percentage of time in drawdown

### Trade Statistics
- **Win Rate**: % of profitable trades
- **Profit Factor**: Total wins / Total losses
- **Expectancy**: Average expected profit per trade
- **Consecutive Wins/Losses**: Longest streaks

### Distribution Metrics
- **Skewness**: Asymmetry of returns distribution
  - Positive: More extreme gains
  - Negative: More extreme losses

- **Kurtosis**: Tail thickness (outlier frequency)
  - High: More outliers

- **VaR (95%)**: Maximum expected loss at 95% confidence
- **CVaR (95%)**: Average loss beyond VaR threshold

### Benchmark Metrics
- **Alpha**: Excess return vs benchmark (risk-adjusted)
- **Beta**: Correlation with benchmark movements
- **Information Ratio**: Consistency of outperformance
- **Correlation**: Linear relationship with benchmark

## Customization

### Themes
Change the dashboard theme by modifying the visualizations theme:

```python
# In app.py
self.visualizations = DashboardVisualizations(theme='plotly_dark')
# Options: 'plotly', 'plotly_dark', 'plotly_white'
```

### Colors
Customize chart colors in `visualizations.py`:

```python
self.colors = {
    'profit': '#00ff00',    # Profit color
    'loss': '#ff0000',      # Loss color
    'neutral': '#4a9eff',   # Neutral/strategy color
    'benchmark': '#ffa500', # Benchmark color
    'grid': '#333333',      # Grid color
}
```

### Metrics
Add custom metrics in `metrics.py`:

```python
def calculate_custom_metric(self, returns: pd.Series) -> float:
    """Calculate your custom metric."""
    # Your calculation here
    return custom_value
```

## Performance Optimization

### Caching
The dashboard uses caching to improve performance:
- Results are cached in memory
- Benchmark data is cached (15-minute expiry)
- Repeated queries don't reload data

### Auto-Refresh
Configure auto-refresh in the sidebar:
1. Enable "Auto Refresh" checkbox
2. Set refresh interval (1-60 seconds)
3. Dashboard updates automatically

### Large Backtests
For backtests with >10,000 trades:
- Consider aggregating trade data
- Use sampling for visualizations
- Increase cache limits

## Troubleshooting

### Dashboard won't start
```bash
# Check if Streamlit is installed
pip install streamlit

# Verify installation
streamlit --version
```

### No data showing
1. Ensure you've run a backtest first
2. Check that results are saved in `/mnt/user-data/outputs/`
3. Verify file permissions
4. Try selecting a different backtest in the sidebar

### Benchmark data not loading
- Requires internet connection
- yfinance must be installed
- Check firewall settings
- Benchmark data is optional (dashboard works without it)

### Performance is slow
- Reduce auto-refresh frequency
- Close unused tabs
- Clear browser cache
- Restart dashboard

## Integration with Backtester

To enable live dashboard updates from your backtester:

```python
# In backtester.py

from dashboard.data_interface import BacktestDataInterface, LiveBacktestStreamer

class Backtester:
    def __init__(self, ...):
        # Add dashboard support
        self.enable_dashboard = True
        self.data_interface = None
        self.streamer = None

    def run(self, ...):
        # Initialize streamer if enabled
        if self.enable_dashboard:
            self.data_interface = BacktestDataInterface(mode='live')
            self.streamer = LiveBacktestStreamer(self.data_interface)
            self.streamer.start()

        # In your main loop
        for date in trading_dates:
            # ... your backtest logic ...

            # Update dashboard
            if self.streamer:
                self.streamer.update(
                    date=date,
                    equity=self.equity,
                    trades=new_trades,
                    positions=self.positions,
                    position_value=self.total_position_value
                )
```

## Export and Sharing

### Export Trade Log
1. Navigate to "Trades" tab
2. Click "📥 Download Trade Log" button
3. CSV file downloads with all trade data

### Export Charts
- Right-click any chart
- Select "Download plot as a png"
- Or use Plotly's built-in export tools

### Share Results
Results are saved as pickle files in `/mnt/user-data/outputs/`:
- `{backtest_name}_results.pkl` - Complete results
- `{backtest_name}_trade_log.csv` - Trade log

## Future Enhancements

- [ ] Strategy comparison mode (compare multiple strategies)
- [ ] Parameter optimization interface
- [ ] Monte Carlo simulation visualization
- [ ] Alert system for live monitoring
- [ ] Custom metric plugins
- [ ] PDF report generation
- [ ] Database integration for large datasets
- [ ] Multi-timeframe analysis
- [ ] Correlation matrix for multi-strategy portfolios

## Support

For issues or questions:
1. Check this README
2. Review the main project documentation
3. Check the example notebooks
4. Review the code comments

## Version History

**v0.1.0** (Current)
- Initial release
- Replay and live modes
- Full metrics suite
- Interactive visualizations
- Benchmark comparison
- Trade analysis

## License

Same as main project license.
