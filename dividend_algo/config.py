"""
Configuration file for Dividend Capture Trading Algorithm
All parameters are modular and can be tuned for optimization
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime

# ============================================================================
# API CREDENTIALS
# ============================================================================

# Alpaca API (Paper Trading)
ALPACA_CONFIG = {
    'API_KEY': 'YOUR_ALPACA_KEY_HERE',
    'SECRET_KEY': 'YOUR_ALPACA_SECRET_HERE',
    'BASE_URL': 'https://paper-api.alpaca.markets'  # Paper trading
}

# ============================================================================
# SYSTEM CONSTANTS
# ============================================================================

# Technical Analysis Constants
TECHNICAL_ANALYSIS_CONSTANTS = {
    'MIN_PRICE_HISTORY_DAYS': 30,  # Minimum days of price history required
    'LOOKBACK_DAYS_DEFAULT': 60,  # Default lookback for technical indicators
    'TRADING_DAYS_PER_YEAR': 252,  # Number of trading days for annualization
}

# Dividend Strategy Constants
DIVIDEND_STRATEGY_CONSTANTS = {
    'EXPECTED_DIV_PRICE_DROP_PCT': 0.70,  # Historical avg: price drops 70% of dividend
    'DIVIDEND_CAPTURE_ALPHA': 0.30,  # Expected capture: 30% of dividend (1 - drop)
    'MEAN_REVERSION_SENSITIVITY': 0.01,  # 1% expected return per std dev from mean
    'MOMENTUM_CONTINUATION_FACTOR': 0.10,  # 10% of recent momentum expected to continue
}

# Data Fetching Constants
DATA_FETCH_CONSTANTS = {
    'PRICE_LOOKBACK_DAYS': 5,  # Days to look back for current price
    'DIVIDEND_HISTORY_WINDOW': 30,  # Days ahead to search for dividends
}

# ============================================================================
# DATA PARAMETERS
# ============================================================================

@dataclass
class DataConfig:
    """Data fetching and processing parameters"""
    
    # Date ranges
    train_start: str = '2018-01-01'
    train_end: str = '2022-12-31'
    test_start: str = '2023-01-01'
    test_end: str = '2024-10-31'
    
    # Lookback periods
    lookback_days: int = 252  # Trading days for historical calculations
    
    # Data sources
    use_alpaca_bars: bool = True
    use_yfinance_fundamentals: bool = True
    
    # Update frequency
    refresh_dividend_calendar: bool = True  # Daily refresh
    refresh_fundamentals: bool = True  # Weekly refresh

# ============================================================================
# STOCK SCREENING PARAMETERS
# ============================================================================

@dataclass
class ScreeningConfig:
    """Primary filters for stock universe selection"""
    
    # Basic filters (must pass all)
    min_dividend_yield: float = 0.02  # 2%
    max_dividend_yield: float = 0.08  # 8%
    min_market_cap: float = 1e9  # $1B
    min_avg_volume: int = 500_000  # shares/day
    min_dividend_history_years: int = 5
    
    # Ex-dividend window
    min_days_to_ex_div: int = 3
    max_days_to_ex_div: int = 20
    
    # Quality metrics thresholds
    min_quality_score: float = 70.0  # Out of 100
    
    # Payout ratio preferences
    optimal_payout_min: float = 0.40  # 40%
    optimal_payout_max: float = 0.60  # 60%
    max_acceptable_payout: float = 0.80  # 80%
    
    # Financial health
    max_debt_to_equity: float = 0.50
    min_roe: float = 0.12  # 12%
    max_pe_ratio: float = 25.0
    
    # Technical filters
    min_rsi: float = 30
    max_rsi: float = 70
    require_positive_momentum: bool = True  # 20-day return > 0
    max_beta: float = 1.2
    max_short_interest: float = 0.10  # 10%

# ============================================================================
# QUALITY SCORING WEIGHTS
# ============================================================================

@dataclass
class ScoringConfig:
    """Weights for multi-factor quality scoring (must sum to 1.0)"""
    
    payout_weight: float = 0.25
    growth_weight: float = 0.25
    financial_weight: float = 0.25
    technical_weight: float = 0.25
    
    # Sub-weights for financial score
    debt_score_weight: float = 0.33
    roe_score_weight: float = 0.33
    pe_score_weight: float = 0.34

# ============================================================================
# ENTRY SIGNAL PARAMETERS
# ============================================================================

@dataclass
class EntryConfig:
    """Entry timing and signal generation"""
    
    # Entry window
    preferred_entry_days: List[int] = None  # Days before ex-div
    
    def __post_init__(self):
        if self.preferred_entry_days is None:
            self.preferred_entry_days = [3, 4, 5]
    
    # Mean reversion filters
    use_z_score_filter: bool = True
    z_score_min: float = -2.0
    z_score_max: float = 0.0  # Slightly oversold
    
    ma_period: int = 20  # Moving average for z-score
    std_period: int = 20  # Standard deviation period
    
    # Volatility filters
    use_volatility_filter: bool = True
    max_realized_vol: float = 0.30  # 30% annualized
    implied_vs_realized_threshold: float = 0.05  # IV > RV + 5%
    
    # Momentum filters
    momentum_period: int = 20  # Days
    require_positive_momentum: bool = True
    
    # Order execution
    use_limit_orders: bool = True
    limit_price_offset: float = 0.001  # 0.1% below bid
    max_order_attempts: int = 3

# ============================================================================
# EXIT SIGNAL PARAMETERS
# ============================================================================

@dataclass
class ExitConfig:
    """Exit rules and profit targets"""
    
    # Primary exit logic
    dividend_adjustment_threshold_low: float = 0.60  # Hold if drop < 60% of div
    dividend_adjustment_threshold_high: float = 0.90  # Exit if drop > 90% of div
    
    # Mean reversion exits (T+3 to T+10)
    min_holding_days: int = 1  # After ex-date
    max_holding_days: int = 10
    
    # Profit targets
    profit_target_multiple: float = 1.5  # 1.5x dividend yield
    profit_target_absolute: float = 0.03  # 3% absolute
    
    # Technical exits
    use_vwap_cross: bool = True
    vwap_period: int = 20
    
    use_entry_plus_dividend: bool = True  # Exit when price >= entry + dividend
    
    # Stop losses
    hard_stop_pct: float = 0.02  # -2%
    use_dividend_stop: bool = True  # Stop = 1x dividend, whichever tighter
    
    trailing_stop_enabled: bool = True
    trailing_stop_activation: float = 0.015  # Activate after +1.5%
    trailing_stop_distance: float = 0.01  # Trail by 1%

# ============================================================================
# POSITION SIZING & RISK MANAGEMENT
# ============================================================================

@dataclass
class RiskConfig:
    """Portfolio-level risk parameters"""
    
    # Kelly Criterion parameters
    use_kelly_sizing: bool = True
    estimated_win_rate: float = 0.60  # 60%
    estimated_win_loss_ratio: float = 2.0  # Avg win / avg loss
    kelly_safety_factor: float = 0.25  # Quarter-Kelly
    
    # Position limits
    max_position_pct: float = 0.02  # 2% of portfolio
    max_positions: int = 25
    min_cash_reserve: float = 0.20  # 20%
    
    # Sector limits
    max_sector_exposure: float = 0.30  # 30% in any sector
    
    # Portfolio constraints
    target_beta_min: float = 0.5
    target_beta_max: float = 0.8
    
    # Risk metrics
    max_portfolio_var_95: float = 0.02  # 2% daily VaR at 95%
    
    # Circuit breakers
    max_daily_loss_pct: float = 0.03  # -3% daily
    max_monthly_loss_pct: float = 0.08  # -8% monthly
    max_drawdown_pct: float = 0.12  # -12% max drawdown
    
    # Correlation management
    max_avg_correlation: float = 0.60
    correlation_window: int = 60  # Days
    
    # Hedging
    use_market_hedge: bool = False  # Disabled by default
    hedge_ratio: float = 0.30  # 30% beta-adjusted notional

# ============================================================================
# MEAN REVERSION MODEL PARAMETERS
# ============================================================================

@dataclass
class MeanReversionConfig:
    """Ornstein-Uhlenbeck and AR(1) parameters"""
    
    use_ou_process: bool = True
    
    # Estimation windows
    ou_estimation_window: int = 60  # Days
    ar1_estimation_window: int = 60
    
    # Thresholds
    min_mean_reversion_speed: float = 0.05  # Minimum theta
    max_mean_reversion_speed: float = 0.50  # Maximum theta
    
    # Long-term mean
    long_term_mean_window: int = 60

# ============================================================================
# BACKTESTING PARAMETERS
# ============================================================================

@dataclass
class BacktestConfig:
    """Backtesting engine configuration"""
    
    # Initial capital
    initial_capital: float = 100_000.0
    
    # Transaction costs
    commission_per_trade: float = 0.0  # Most brokers are zero
    slippage_bps: float = 5  # 0.05% per trade
    sec_fee_per_dollar: float = 0.0000278  # SEC fees
    
    # Execution assumptions
    assume_market_impact: bool = True
    market_impact_factor: float = 0.10  # 10% of ADV liquidity impact
    
    # Rebalancing
    rebalance_on_new_signals: bool = True
    max_trades_per_day: int = 10
    
    # Walk-forward analysis
    use_walk_forward: bool = True
    walk_forward_train_days: int = 252 * 2  # 2 years training
    walk_forward_test_days: int = 252 // 2  # 6 months testing
    walk_forward_step_days: int = 252 // 4  # 3 months step

# ============================================================================
# ANALYTICS & REPORTING
# ============================================================================

@dataclass
class AnalyticsConfig:
    """Performance analytics and visualization"""
    
    # Benchmark
    benchmark_ticker: str = 'SPY'
    
    # Risk-free rate
    risk_free_rate: float = 0.045  # 4.5% annual
    
    # Plotting
    plot_equity_curve: bool = True
    plot_drawdown: bool = True
    plot_monthly_returns: bool = True
    plot_rolling_metrics: bool = True
    plot_correlation_matrix: bool = True
    
    # Rolling windows for metrics
    rolling_sharpe_window: int = 252  # 1 year
    rolling_beta_window: int = 60  # 3 months
    
    # Report generation
    generate_html_report: bool = True
    save_trade_log: bool = True
    
    # Performance metrics to calculate
    calculate_metrics: List[str] = None
    
    def __post_init__(self):
        if self.calculate_metrics is None:
            self.calculate_metrics = [
                'total_return',
                'annual_return',
                'sharpe_ratio',
                'sortino_ratio',
                'calmar_ratio',
                'max_drawdown',
                'win_rate',
                'profit_factor',
                'avg_win',
                'avg_loss',
                'avg_holding_period',
                'total_trades',
                'beta',
                'alpha',
                'var_95',
                'cvar_95'
            ]

# ============================================================================
# MONTE CARLO SIMULATION
# ============================================================================

@dataclass
class MonteCarloConfig:
    """Monte Carlo validation parameters"""
    
    enabled: bool = True
    num_simulations: int = 1000
    confidence_level: float = 0.95
    
    # Bootstrap parameters
    block_size: int = 20  # Days per block for time-series bootstrap
    resample_with_replacement: bool = True

# ============================================================================
# INSTANTIATE DEFAULT CONFIGS
# ============================================================================

# Create default configuration objects
data_config = DataConfig()
screening_config = ScreeningConfig()
scoring_config = ScoringConfig()
entry_config = EntryConfig()
exit_config = ExitConfig()
risk_config = RiskConfig()
mean_reversion_config = MeanReversionConfig()
backtest_config = BacktestConfig()
analytics_config = AnalyticsConfig()
monte_carlo_config = MonteCarloConfig()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_config():
    """Validate that all configurations are consistent"""
    errors = []
    warnings = []

    # Check API keys
    if ALPACA_CONFIG['API_KEY'] == 'YOUR_ALPACA_KEY_HERE':
        warnings.append("Alpaca API keys not configured (using default placeholder)")
    if ALPACA_CONFIG['SECRET_KEY'] == 'YOUR_ALPACA_SECRET_HERE':
        warnings.append("Alpaca secret key not configured (using default placeholder)")

    # Check scoring weights sum to 1.0
    total_weight = (scoring_config.payout_weight +
                   scoring_config.growth_weight +
                   scoring_config.financial_weight +
                   scoring_config.technical_weight)
    if abs(total_weight - 1.0) > 0.01:
        errors.append(f"Scoring weights sum to {total_weight}, must equal 1.0")

    # Check date ranges
    if data_config.train_end >= data_config.test_start:
        errors.append("Train end date must be before test start date")

    # Check risk parameters
    if risk_config.max_position_pct * risk_config.max_positions > 1.0:
        errors.append("Max positions could exceed 100% of portfolio")

    # Return both errors and warnings
    return {'errors': errors, 'warnings': warnings}

def print_config_summary():
    """Print a summary of current configuration"""
    print("=" * 80)
    print("DIVIDEND CAPTURE ALGORITHM - CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"\n📅 Training Period: {data_config.train_start} to {data_config.train_end}")
    print(f"📅 Testing Period: {data_config.test_start} to {data_config.test_end}")
    print(f"\n💰 Initial Capital: ${backtest_config.initial_capital:,.0f}")
    print(f"💰 Max Position Size: {risk_config.max_position_pct*100:.1f}%")
    print(f"💰 Max Positions: {risk_config.max_positions}")
    print(f"\n📊 Screening Criteria:")
    print(f"   - Dividend Yield: {screening_config.min_dividend_yield*100:.1f}% - {screening_config.max_dividend_yield*100:.1f}%")
    print(f"   - Min Market Cap: ${screening_config.min_market_cap/1e9:.1f}B")
    print(f"   - Min Quality Score: {screening_config.min_quality_score}/100")
    print(f"\n⚡ Entry Window: {entry_config.preferred_entry_days} days before ex-div")
    print(f"⚡ Max Holding Period: {exit_config.max_holding_days} days")
    print(f"\n🛡️ Risk Management:")
    print(f"   - Hard Stop Loss: {risk_config.max_daily_loss_pct*100:.1f}%")
    print(f"   - Max Drawdown: {risk_config.max_drawdown_pct*100:.1f}%")
    print(f"   - Kelly Safety Factor: {risk_config.kelly_safety_factor}")
    print(f"\n📈 Performance Targets:")
    print(f"   - Target Sharpe: 1.5+")
    print(f"   - Target Annual Return: 10-15%")
    print(f"   - Target Win Rate: 55%+")
    print("=" * 80)

if __name__ == '__main__':
    # Validate configuration
    validation = validate_config()

    if validation['errors']:
        print("❌ Configuration Errors:")
        for error in validation['errors']:
            print(f"   - {error}")

    if validation['warnings']:
        print("\n⚠️  Configuration Warnings:")
        for warning in validation['warnings']:
            print(f"   - {warning}")

    if not validation['errors']:
        print("\n✅ Configuration validated successfully")
        print_config_summary()
