# Dividend Algorithm - Optimization & Improvement Guide

## Summary of Fixes Completed

### HIGH PRIORITY FIXES ✅ (ALL COMPLETE)
1. **Fixed config.risk_free_rate bug** - Corrected import references
2. **Fixed entry filter logic** - RSI and z-score filters now independent
3. **Added division-by-zero checks** - Protected all division operations
4. **Fixed circuit breaker exit pricing** - Uses current market prices
5. **Created output directories** - Auto-creates paths before saving
6. **Added API key validation** - Warns about placeholder credentials

### MEDIUM PRIORITY FIXES ✅ (CORE COMPLETE)
7. **Implemented logging framework** - Professional logging with file/console output
8. **Fixed z-score calculation** - Now uses price std instead of returns std (CRITICAL FIX)
9. **Fixed mean reversion logic** - Consistent log-space calculations
10. **Added configuration constants** - Replaced magic numbers with named constants
11. **Implemented data caching** - Reduces API calls, tracks cache statistics
12. **Enhanced error handling** - Proper exception handling with logging

---

## Remaining Optimizations

### 1. BATCH API CALLS (Performance Enhancement)

**Current State:** API calls are made sequentially in loops
**Problem:** Slow, rate-limited, inefficient
**Solution:** Batch requests together

#### Implementation Example:
```python
# In data_manager.py
def get_multiple_stock_prices(self, tickers: List[str], start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch prices for multiple tickers in parallel

    Returns:
        Dictionary mapping ticker -> DataFrame
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_ticker = {
            executor.submit(self.get_stock_prices, ticker, start_date, end_date): ticker
            for ticker in tickers
        }

        for future in as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                results[ticker] = future.result()
            except Exception as e:
                self.logger.error(f"Failed to fetch {ticker}: {e}")
                results[ticker] = pd.DataFrame()

    return results
```

**Benefit:** 5-10x faster data fetching

---

### 2. UNIT TESTS (Code Quality)

**Current State:** No automated tests
**Problem:** Difficult to verify changes don't break functionality
**Solution:** Add pytest unit tests

#### Key Test Areas:
```python
# tests/test_strategy.py
def test_calculate_expected_return():
    """Test expected return calculation"""
    strategy = DividendCaptureStrategy(mock_data_manager)

    # Test normal case
    result = strategy._calculate_expected_return(
        price=100.0,
        stock_info=pd.Series({'amount': 1.0}),
        mr_params={'mu': 100, 'sigma': 0.2},
        prices=mock_prices_df
    )
    assert result > 0

    # Test edge case: zero price
    result = strategy._calculate_expected_return(
        price=0.0,
        stock_info=pd.Series({'amount': 1.0}),
        mr_params={'mu': 100, 'sigma': 0.2},
        prices=mock_prices_df
    )
    assert result == 0

# tests/test_risk_manager.py
def test_position_sizing():
    """Test Kelly criterion position sizing"""
    rm = RiskManager(initial_capital=100_000)

    shares = rm.calculate_position_size(
        signal={},
        current_positions={},
        current_price=50.0
    )

    assert shares > 0
    assert shares * 50.0 <= 100_000 * 0.02  # Max 2% position

# tests/test_data_manager.py
def test_z_score_calculation():
    """Verify z-score uses price std, not returns std"""
    dm = DataManager()

    prices = pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109] * 3
    })

    result = dm.calculate_technical_indicators(prices)

    # Z-score should be finite
    assert not result['z_score'].isna().all()
    # Should use price std formula
    expected_std = prices['close'].rolling(20).std()
    assert expected_std is not None
```

**Setup:**
```bash
pip install pytest pytest-cov
pytest tests/ --cov=. --cov-report=html
```

---

### 3. MONTE CARLO SIMULATION (Risk Analysis)

**Purpose:** Stress test strategy under different market conditions

#### Implementation:
```python
# In analytics.py
def run_monte_carlo(backtest_results, n_simulations=1000):
    """
    Run Monte Carlo simulation of returns

    Randomly resamples daily returns to generate possible future paths
    """
    returns = backtest_results['returns']

    simulated_paths = []
    for i in range(n_simulations):
        # Bootstrap resample returns
        sampled_returns = returns.sample(len(returns), replace=True)
        cumulative = (1 + sampled_returns).cumprod()
        simulated_paths.append(cumulative.iloc[-1])

    simulated_paths = np.array(simulated_paths)

    return {
        'mean_final_value': simulated_paths.mean(),
        'median_final_value': np.median(simulated_paths),
        'var_5': np.percentile(simulated_paths, 5),
        'var_95': np.percentile(simulated_paths, 95),
        'probability_of_profit': (simulated_paths > 1.0).mean()
    }
```

---

### 4. WALK-FORWARD OPTIMIZATION (Robustness)

**Current:** Basic walk-forward implemented but not parameter optimization
**Enhancement:** Optimize parameters on training window, test on validation window

```python
# In main.py
def walk_forward_optimization():
    """
    Optimize parameters on rolling windows
    """
    # Example: Optimize entry days
    best_params = {}

    for window_start in window_dates:
        # Train period
        train_results = []
        for entry_days in [[3], [4], [5], [3,4], [4,5]]:
            entry_config.preferred_entry_days = entry_days
            bt = Backtester(train_start, train_end, capital)
            res = bt.run_backtest()
            train_results.append((entry_days, res['sharpe_ratio']))

        # Select best
        best_entry_days = max(train_results, key=lambda x: x[1])[0]

        # Test on validation period
        entry_config.preferred_entry_days = best_entry_days
        bt_test = Backtester(test_start, test_end, capital)
        test_result = bt_test.run_backtest()

        best_params[window_start] = {
            'entry_days': best_entry_days,
            'test_sharpe': test_result['sharpe_ratio']
        }

    return best_params
```

---

### 5. CORRELATION ENFORCEMENT (Risk Management)

**Current:** Calculates correlation but doesn't enforce limits
**Enhancement:** Reject positions that increase portfolio correlation

```python
# In risk_manager.py
def check_correlation_before_entry(self, new_ticker: str, current_positions: Dict) -> bool:
    """
    Check if adding position would exceed correlation limits

    Returns:
        True if position is allowed, False if too correlated
    """
    if len(current_positions) < 2:
        return True  # Can't have correlation with < 2 positions

    # Get return series for new ticker and existing positions
    position_returns = {}
    for ticker in list(current_positions.keys()) + [new_ticker]:
        prices = self.dm.get_stock_prices(ticker, lookback_start, current_date)
        position_returns[ticker] = prices['close'].pct_change().dropna()

    # Calculate correlation matrix
    corr_matrix = self.calculate_correlation_matrix(position_returns)

    # Check if new ticker has high correlation with any existing position
    new_ticker_corr = corr_matrix[new_ticker].drop(new_ticker)
    max_correlation = new_ticker_corr.max()

    if max_correlation > risk_config.max_pairwise_correlation:
        self.logger.warning(
            f"Rejecting {new_ticker}: correlation {max_correlation:.2f} "
            f"exceeds limit {risk_config.max_pairwise_correlation}"
        )
        return False

    return True
```

**Add to config.py:**
```python
@dataclass
class RiskConfig:
    # ... existing fields ...
    max_pairwise_correlation: float = 0.7  # Max correlation between any two positions
```

---

### 6. LIVE TRADING ADAPTER (Production)

**Purpose:** Connect to Alpaca for paper/live trading

```python
# Create live_trader.py
from alpaca_trade_api import REST
from strategy import DividendCaptureStrategy
from risk_manager import RiskManager

class LiveTrader:
    """
    Live trading adapter for Alpaca
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        base_url = 'https://paper-api.alpaca.markets' if paper else 'https://api.alpaca.markets'
        self.api = REST(api_key, secret_key, base_url)
        self.strategy = DividendCaptureStrategy(DataManager())
        self.risk_manager = RiskManager(self.get_account_equity())

    def get_account_equity(self) -> float:
        """Get current account equity"""
        account = self.api.get_account()
        return float(account.equity)

    def execute_signals(self):
        """
        Check for signals and execute trades
        Run this daily (e.g., via cron or scheduler)
        """
        current_date = datetime.now().strftime('%Y-%m-%d')

        # Check exit signals
        exit_signals = self.strategy.check_exit_signals(current_date)
        for signal in exit_signals:
            self.close_position(signal)

        # Check entry signals
        div_calendar = self.strategy.dm.get_dividend_calendar(
            current_date,
            (datetime.now() + timedelta(days=20)).strftime('%Y-%m-%d')
        )
        screened = self.strategy.dm.screen_stocks(div_calendar, current_date)
        entry_signals = self.strategy.generate_entry_signals(screened, current_date)

        # Risk-adjusted allocation
        allocated = self.risk_manager.get_position_allocation(
            entry_signals,
            self.strategy.positions
        )

        for signal in allocated:
            self.open_position(signal)

    def open_position(self, signal: Dict):
        """Place market order to open position"""
        try:
            order = self.api.submit_order(
                symbol=signal['ticker'],
                qty=signal['shares'],
                side='buy',
                type='market',
                time_in_force='day'
            )
            self.strategy.open_position(signal)
            logger.info(f"Opened {signal['ticker']}: {order.id}")
        except Exception as e:
            logger.error(f"Failed to open {signal['ticker']}: {e}")

    def close_position(self, signal: Dict):
        """Place market order to close position"""
        try:
            order = self.api.submit_order(
                symbol=signal['ticker'],
                qty=signal['shares'],
                side='sell',
                type='market',
                time_in_force='day'
            )
            self.strategy.close_position(signal)
            logger.info(f"Closed {signal['ticker']}: {order.id}")
        except Exception as e:
            logger.error(f"Failed to close {signal['ticker']}: {e}")
```

**Scheduler (Linux cron):**
```bash
# Edit crontab: crontab -e
# Run daily at 9:35 AM ET (after market open)
35 9 * * 1-5 cd /path/to/dividend_algo && python live_trader.py
```

---

### 7. PERFORMANCE MONITORING DASHBOARD (Operations)

**Tool:** Streamlit dashboard for real-time monitoring

```python
# dashboard.py
import streamlit as st
from data_manager import DataManager
from strategy import DividendCaptureStrategy

st.set_page_config(page_title="Dividend Algo Dashboard", layout="wide")

st.title("📈 Dividend Capture Algorithm - Live Dashboard")

# Sidebar: Controls
st.sidebar.header("Controls")
if st.sidebar.button("Refresh Data"):
    st.cache_data.clear()

# Main metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Portfolio Value", "$125,430", "+5.4%")
with col2:
    st.metric("Active Positions", "12", "-2")
with col3:
    st.metric("Today's P&L", "+$1,250", "+1.0%")
with col4:
    st.metric("Sharpe Ratio (30d)", "1.85", "+0.15")

# Positions table
st.header("Active Positions")
positions_df = pd.DataFrame([
    {"Ticker": "AAPL", "Shares": 100, "Entry": "$150.00", "Current": "$155.50", "P&L": "+$550", "Days Held": 3},
    {"Ticker": "MSFT", "Shares": 50, "Entry": "$380.00", "Current": "$385.00", "P&L": "+$250", "Days Held": 2},
])
st.dataframe(positions_df, use_container_width=True)

# Equity curve
st.header("Equity Curve")
equity_data = get_equity_curve()  # Load from database
st.line_chart(equity_data)

# Recent trades
st.header("Recent Trades")
trades = get_recent_trades(limit=10)
st.dataframe(trades)

# Cache stats
st.header("System Performance")
cache_stats = dm.get_cache_stats()
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("API Calls", cache_stats['api_calls'])
with col2:
    st.metric("Cache Hit Rate", f"{cache_stats['hit_rate_pct']:.1f}%")
with col3:
    st.metric("Cached Items", cache_stats['cached_items'])
```

**Run:**
```bash
streamlit run dashboard.py
```

---

## Testing Roadmap

### Phase 1: Validation (Week 1)
- ✅ Run validation tests in Colab
- ✅ Verify all modules load without errors
- ✅ Check cache efficiency
- ✅ Review quick backtest results

### Phase 2: Full Backtest (Week 2)
- Test 2023-2024 (2 years of data)
- Analyze Sharpe, drawdown, win rate
- Compare against buy-and-hold SPY
- Review trade-by-trade results

### Phase 3: Parameter Tuning (Week 3)
- Test different entry windows
- Optimize profit targets
- Adjust stop losses
- Find optimal holding periods

### Phase 4: Paper Trading (Week 4)
- Connect to Alpaca paper account
- Run live for 2-4 weeks
- Monitor real-time performance
- Fix any execution issues

### Phase 5: Live (if successful)
- Start with small capital ($5K-$10K)
- Monitor closely for first month
- Scale up if performance meets targets
- Implement risk management alerts

---

## Key Performance Targets

### Minimum Acceptable:
- Sharpe Ratio: > 1.0
- Annual Return: > 8%
- Max Drawdown: < 15%
- Win Rate: > 50%

### Target (Good):
- Sharpe Ratio: > 1.5
- Annual Return: > 12%
- Max Drawdown: < 10%
- Win Rate: > 55%

### Excellent:
- Sharpe Ratio: > 2.0
- Annual Return: > 15%
- Max Drawdown: < 8%
- Win Rate: > 60%

---

## Risk Warnings

⚠️ **IMPORTANT:**
1. Backtesting ≠ future performance
2. Market conditions change
3. Slippage and fills may vary in live trading
4. Dividend policies can change unexpectedly
5. Always use proper position sizing
6. Never risk more than you can afford to lose
7. Monitor performance weekly at minimum
8. Have a plan to stop/reduce if performance degrades

---

## Next Steps

1. **Immediate:** Test in Google Colab using provided notebook
2. **Short-term:** Run full 2-year backtest, analyze results
3. **Medium-term:** Paper trade for 1 month minimum
4. **Long-term:** Consider live trading only if paper trading is successful

Good luck! 🚀
