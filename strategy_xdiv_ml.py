"""
Enhanced X-Dividend Strategy with Machine Learning Training Period
This strategy learns optimal parameters from historical data during training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

from config import (
    entry_config, exit_config, risk_config, screening_config,
    TECHNICAL_ANALYSIS_CONSTANTS, DIVIDEND_STRATEGY_CONSTANTS
)
from data_manager import DataManager


class XDividendMLStrategy:
    """
    Enhanced dividend strategy with:
    1. Training period to learn optimal parameters
    2. Adaptive entry/exit thresholds based on historical performance
    3. Stock-specific dividend capture rates
    4. Dynamic position sizing based on confidence
    """

    def __init__(self, data_manager: DataManager, use_training: bool = True):
        self.dm = data_manager
        self.use_training = use_training
        self.positions = {}
        self.pending_orders = {}
        self.position_history = []

        # Learned parameters (initialized with defaults, updated during training)
        self.learned_params = {
            'avg_dividend_capture_rate': 0.30,  # Will be learned
            'optimal_entry_days': [3, 4, 5],     # Will be optimized
            'optimal_hold_days': 7,               # Will be optimized
            'z_score_threshold': -1.0,            # Will be optimized
            'rsi_threshold_low': 35,              # Will be optimized
            'rsi_threshold_high': 65,             # Will be optimized
            'min_expected_return': 0.015,         # Will be optimized (1.5%)
            'stock_specific_rates': {},           # Ticker -> capture rate
            'sector_performance': {},             # Sector -> avg return
            'volatility_adjustment': 1.0,         # Volatility multiplier
        }

        # Training metrics
        self.training_completed = False
        self.training_metrics = {}

    def train(self, start_date: str, end_date: str, verbose: bool = True):
        """
        Train the strategy on historical data to learn optimal parameters

        Args:
            start_date: Training period start (YYYY-MM-DD)
            end_date: Training period end (YYYY-MM-DD)
            verbose: Print training progress
        """
        if verbose:
            print("\n" + "="*80)
            print("🎓 TRAINING X-DIVIDEND ML STRATEGY")
            print("="*80)
            print(f"Training Period: {start_date} to {end_date}")
            print("="*80 + "\n")

        # Step 1: Collect historical dividend events and outcomes
        if verbose:
            print("📊 Step 1/4: Collecting historical dividend data...")

        div_calendar = self.dm.get_dividend_calendar(start_date, end_date, lookback_days=30)

        if len(div_calendar) == 0:
            print("⚠️  No dividend data available for training period")
            return

        # Step 2: Analyze ex-dividend price behavior
        if verbose:
            print(f"📊 Step 2/4: Analyzing {len(div_calendar)} dividend events...")

        ex_div_analysis = self._analyze_ex_dividend_behavior(div_calendar, start_date, end_date)

        # Step 3: Optimize entry/exit parameters
        if verbose:
            print("📊 Step 3/4: Optimizing entry/exit parameters...")

        optimal_params = self._optimize_parameters(ex_div_analysis)

        # Step 4: Learn stock-specific patterns
        if verbose:
            print("📊 Step 4/4: Learning stock-specific patterns...")

        stock_patterns = self._learn_stock_patterns(ex_div_analysis)

        # Update learned parameters
        self.learned_params.update(optimal_params)
        self.learned_params['stock_specific_rates'] = stock_patterns['capture_rates']
        self.learned_params['sector_performance'] = stock_patterns['sector_performance']

        # Store training metrics
        self.training_metrics = {
            'training_period': f"{start_date} to {end_date}",
            'num_dividend_events': len(div_calendar),
            'num_analyzed_events': len(ex_div_analysis),
            'avg_capture_rate': self.learned_params['avg_dividend_capture_rate'],
            'optimal_entry_days': self.learned_params['optimal_entry_days'],
            'optimal_hold_days': self.learned_params['optimal_hold_days'],
            'learned_parameters': self.learned_params
        }

        self.training_completed = True

        if verbose:
            self._print_training_summary()

    def _analyze_ex_dividend_behavior(self, div_calendar: pd.DataFrame,
                                     start_date: str, end_date: str) -> List[Dict]:
        """
        Analyze how stocks behave around ex-dividend dates
        Returns list of analyzed dividend events with actual outcomes
        """
        analysis_results = []

        for idx, row in div_calendar.iterrows():
            ticker = row['ticker']
            ex_date = row['ex_date']
            div_amount = row['amount']

            # Skip if ex_date is outside our date range
            ex_date_dt = pd.to_datetime(ex_date).tz_localize(None)
            if ex_date_dt < pd.to_datetime(start_date).tz_localize(None):
                continue
            if ex_date_dt > pd.to_datetime(end_date).tz_localize(None):
                continue

            # Get price data around ex-dividend date
            lookback_start = (ex_date_dt - timedelta(days=30)).strftime('%Y-%m-%d')
            lookforward_end = (ex_date_dt + timedelta(days=20)).strftime('%Y-%m-%d')

            try:
                prices = self.dm.get_stock_prices(ticker, lookback_start, lookforward_end)

                if len(prices) < 20:
                    continue

                # Calculate technical indicators
                prices = self.dm.calculate_technical_indicators(prices)

                # Find price at different points
                pre_ex_idx = prices.index.get_indexer([ex_date_dt], method='ffill')[0]
                if pre_ex_idx < 1:
                    continue

                pre_ex_price = prices['close'].iloc[pre_ex_idx - 1]  # Day before ex-div

                # Find ex-div day price
                if pre_ex_idx >= len(prices):
                    continue
                ex_div_price = prices['close'].iloc[pre_ex_idx]

                # Calculate actual drop
                actual_drop = pre_ex_price - ex_div_price
                expected_drop = div_amount

                if expected_drop == 0:
                    continue

                capture_rate = 1 - (actual_drop / expected_drop) if expected_drop > 0 else 0

                # Calculate returns at different holding periods (1, 3, 5, 7, 10 days post ex-div)
                returns_by_hold_period = {}
                for hold_days in [1, 3, 5, 7, 10]:
                    exit_idx = pre_ex_idx + hold_days
                    if exit_idx < len(prices):
                        exit_price = prices['close'].iloc[exit_idx]
                        # Return from day before ex-div to exit
                        hold_return = (exit_price - pre_ex_price + div_amount) / pre_ex_price
                        returns_by_hold_period[hold_days] = hold_return

                # Get technical indicators at entry (3-5 days before ex-div)
                for entry_offset in [3, 4, 5]:
                    entry_idx = pre_ex_idx - entry_offset
                    if entry_idx >= 0 and entry_idx < len(prices):
                        entry_price = prices['close'].iloc[entry_idx]

                        analysis_results.append({
                            'ticker': ticker,
                            'ex_date': ex_date,
                            'sector': row.get('sector', 'Unknown'),
                            'dividend_amount': div_amount,
                            'dividend_yield': div_amount / pre_ex_price,
                            'entry_offset_days': entry_offset,
                            'entry_price': entry_price,
                            'pre_ex_price': pre_ex_price,
                            'ex_div_price': ex_div_price,
                            'actual_drop': actual_drop,
                            'capture_rate': capture_rate,
                            'returns_by_hold_period': returns_by_hold_period,
                            'rsi_at_entry': prices['rsi_14'].iloc[entry_idx] if entry_idx < len(prices) else np.nan,
                            'z_score_at_entry': prices['z_score'].iloc[entry_idx] if entry_idx < len(prices) else np.nan,
                            'volatility_at_entry': prices['volatility_20'].iloc[entry_idx] if entry_idx < len(prices) else np.nan,
                        })

            except Exception as e:
                # Skip problematic tickers
                continue

        return analysis_results

    def _optimize_parameters(self, analysis_results: List[Dict]) -> Dict:
        """
        Optimize strategy parameters based on historical analysis
        """
        if len(analysis_results) == 0:
            return {}

        df = pd.DataFrame(analysis_results)

        # 1. Calculate average capture rate
        avg_capture_rate = df['capture_rate'].median()
        avg_capture_rate = max(0.1, min(0.8, avg_capture_rate))  # Bound between 10-80%

        # 2. Find optimal entry days (which offset has best returns)
        entry_performance = {}
        for offset in [3, 4, 5]:
            offset_data = df[df['entry_offset_days'] == offset]
            if len(offset_data) > 0:
                # Average best return across all holding periods
                avg_returns = []
                for _, row in offset_data.iterrows():
                    if row['returns_by_hold_period']:
                        best_return = max(row['returns_by_hold_period'].values())
                        avg_returns.append(best_return)

                if avg_returns:
                    entry_performance[offset] = np.mean(avg_returns)

        # Select top performing entry days
        sorted_offsets = sorted(entry_performance.items(), key=lambda x: x[1], reverse=True)
        optimal_entry_days = [offset for offset, _ in sorted_offsets[:3]]
        if not optimal_entry_days:
            optimal_entry_days = [3, 4, 5]

        # 3. Find optimal holding period
        hold_period_returns = {1: [], 3: [], 5: [], 7: [], 10: []}
        for _, row in df.iterrows():
            for hold_days, ret in row['returns_by_hold_period'].items():
                hold_period_returns[hold_days].append(ret)

        # Calculate Sharpe-like metric for each holding period
        best_hold_period = 7
        best_sharpe = -999
        for hold_days, returns in hold_period_returns.items():
            if len(returns) > 5:
                mean_ret = np.mean(returns)
                std_ret = np.std(returns)
                sharpe = mean_ret / std_ret if std_ret > 0 else 0
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_hold_period = hold_days

        # 4. Optimize RSI thresholds (find RSI range with best returns)
        rsi_valid = df[df['rsi_at_entry'].notna()]
        if len(rsi_valid) > 10:
            # Try different RSI ranges
            best_rsi_range = (35, 65)
            best_avg_return = -999

            for rsi_low in range(25, 45, 5):
                for rsi_high in range(60, 75, 5):
                    rsi_subset = rsi_valid[
                        (rsi_valid['rsi_at_entry'] >= rsi_low) &
                        (rsi_valid['rsi_at_entry'] <= rsi_high)
                    ]
                    if len(rsi_subset) > 5:
                        # Get best returns for this RSI range
                        avg_returns = []
                        for _, row in rsi_subset.iterrows():
                            if row['returns_by_hold_period']:
                                avg_returns.append(max(row['returns_by_hold_period'].values()))

                        if avg_returns and np.mean(avg_returns) > best_avg_return:
                            best_avg_return = np.mean(avg_returns)
                            best_rsi_range = (rsi_low, rsi_high)

            rsi_threshold_low, rsi_threshold_high = best_rsi_range
        else:
            rsi_threshold_low, rsi_threshold_high = 35, 65

        # 5. Optimize z-score threshold
        z_valid = df[df['z_score_at_entry'].notna()]
        if len(z_valid) > 10:
            # Find z-score threshold that maximizes returns
            best_z_threshold = -1.0
            best_avg_return = -999

            for z_thresh in np.arange(-2.0, 0.5, 0.25):
                z_subset = z_valid[z_valid['z_score_at_entry'] <= z_thresh]
                if len(z_subset) > 5:
                    avg_returns = []
                    for _, row in z_subset.iterrows():
                        if row['returns_by_hold_period']:
                            avg_returns.append(max(row['returns_by_hold_period'].values()))

                    if avg_returns and np.mean(avg_returns) > best_avg_return:
                        best_avg_return = np.mean(avg_returns)
                        best_z_threshold = z_thresh

            z_score_threshold = best_z_threshold
        else:
            z_score_threshold = -1.0

        # 6. Calculate minimum expected return threshold (25th percentile of positive returns)
        all_returns = []
        for _, row in df.iterrows():
            if row['returns_by_hold_period']:
                all_returns.extend(row['returns_by_hold_period'].values())

        positive_returns = [r for r in all_returns if r > 0]
        if len(positive_returns) > 10:
            min_expected_return = np.percentile(positive_returns, 25)
        else:
            min_expected_return = 0.01  # 1%

        return {
            'avg_dividend_capture_rate': avg_capture_rate,
            'optimal_entry_days': optimal_entry_days,
            'optimal_hold_days': best_hold_period,
            'z_score_threshold': z_score_threshold,
            'rsi_threshold_low': rsi_threshold_low,
            'rsi_threshold_high': rsi_threshold_high,
            'min_expected_return': min_expected_return,
        }

    def _learn_stock_patterns(self, analysis_results: List[Dict]) -> Dict:
        """
        Learn stock-specific and sector-specific patterns
        """
        if len(analysis_results) == 0:
            return {'capture_rates': {}, 'sector_performance': {}}

        df = pd.DataFrame(analysis_results)

        # Stock-specific capture rates (minimum 3 observations)
        stock_capture_rates = {}
        for ticker in df['ticker'].unique():
            ticker_data = df[df['ticker'] == ticker]
            if len(ticker_data) >= 3:
                median_capture = ticker_data['capture_rate'].median()
                stock_capture_rates[ticker] = median_capture

        # Sector performance
        sector_performance = {}
        for sector in df['sector'].unique():
            sector_data = df[df['sector'] == sector]
            if len(sector_data) >= 5:
                # Calculate average return for best holding period
                avg_returns = []
                for _, row in sector_data.iterrows():
                    if row['returns_by_hold_period']:
                        avg_returns.append(max(row['returns_by_hold_period'].values()))

                if avg_returns:
                    sector_performance[sector] = np.mean(avg_returns)

        return {
            'capture_rates': stock_capture_rates,
            'sector_performance': sector_performance
        }

    def _print_training_summary(self):
        """Print summary of training results"""
        print("\n" + "="*80)
        print("✅ TRAINING COMPLETED")
        print("="*80)
        print(f"\n📊 Training Metrics:")
        print(f"  Dividend Events Analyzed: {self.training_metrics['num_analyzed_events']}")
        print(f"  Average Capture Rate:     {self.learned_params['avg_dividend_capture_rate']*100:.1f}%")
        print(f"  Optimal Entry Days:       {self.learned_params['optimal_entry_days']}")
        print(f"  Optimal Hold Period:      {self.learned_params['optimal_hold_days']} days")
        print(f"  Z-Score Threshold:        {self.learned_params['z_score_threshold']:.2f}")
        print(f"  RSI Range:                {self.learned_params['rsi_threshold_low']:.0f} - {self.learned_params['rsi_threshold_high']:.0f}")
        print(f"  Min Expected Return:      {self.learned_params['min_expected_return']*100:.2f}%")
        print(f"  Stock-Specific Learned:   {len(self.learned_params['stock_specific_rates'])} tickers")
        print(f"  Sector Patterns Learned:  {len(self.learned_params['sector_performance'])} sectors")
        print("="*80 + "\n")

    def generate_entry_signals(self, screened_stocks: pd.DataFrame,
                               current_date: str) -> List[Dict]:
        """
        Generate entry signals using learned parameters
        """
        if len(screened_stocks) == 0:
            return []

        signals = []
        current_dt = pd.to_datetime(current_date).tz_localize(None)

        print(f"\n🎯 Generating entry signals for {len(screened_stocks)} candidates...")

        if self.training_completed:
            print(f"   Using learned parameters (capture rate: {self.learned_params['avg_dividend_capture_rate']*100:.1f}%)")

        entry_filter_failures = {}

        for idx, row in screened_stocks.iterrows():
            ticker = row['ticker']

            # Get recent price data
            lookback_days = TECHNICAL_ANALYSIS_CONSTANTS['LOOKBACK_DAYS_DEFAULT']
            lookback_start = (current_dt - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            prices = self.dm.get_stock_prices(ticker, lookback_start, current_date)

            if len(prices) < TECHNICAL_ANALYSIS_CONSTANTS['MIN_PRICE_HISTORY_DAYS']:
                entry_filter_failures[ticker] = f"Insufficient history ({len(prices)} days)"
                continue

            # Calculate technical indicators
            prices = self.dm.calculate_technical_indicators(prices)

            # Apply learned entry filters
            passed, reason = self._passes_learned_entry_filters(prices, row)
            if not passed:
                entry_filter_failures[ticker] = reason.split(': ', 1)[1] if ': ' in reason else reason
                continue

            # Calculate entry price
            current_price = prices['close'].iloc[-1]

            # Calculate expected return using learned parameters
            expected_return = self._calculate_learned_expected_return(
                current_price, row, prices
            )

            # Filter by learned minimum expected return
            if expected_return < self.learned_params['min_expected_return']:
                entry_filter_failures[ticker] = f"Expected return {expected_return*100:.2f}% < {self.learned_params['min_expected_return']*100:.2f}%"
                continue

            # Build signal
            signal = {
                'ticker': ticker,
                'entry_date': current_date,
                'entry_price': current_price,
                'ex_div_date': row['ex_date'],
                'days_to_ex_div': row['days_to_ex_div'],
                'dividend_amount': row['amount'],
                'dividend_yield': row['yield'],
                'quality_score': row['quality_score'],
                'expected_return': expected_return,
                'sector': row.get('sector', 'Unknown'),

                # Technical context
                'rsi': prices['rsi_14'].iloc[-1],
                'z_score': prices['z_score'].iloc[-1],
                'momentum_20': prices['momentum_20'].iloc[-1],
                'volatility_20': prices['volatility_20'].iloc[-1],

                # Learned parameters
                'learned_capture_rate': self._get_stock_capture_rate(ticker),
                'target_hold_days': self.learned_params['optimal_hold_days'],

                # Stop loss
                'stop_loss': self._calculate_stop_loss(current_price, row['amount']),
            }

            signals.append(signal)

        # Sort by expected return
        signals = sorted(signals, key=lambda x: x['expected_return'], reverse=True)

        print(f"✅ Generated {len(signals)} entry signals")

        # Show failures
        if len(signals) == 0 and entry_filter_failures:
            print("\n📋 Entry filter failures:")
            for ticker, reason in list(entry_filter_failures.items())[:10]:
                print(f"   {ticker}: {reason}")

        return signals

    def _passes_learned_entry_filters(self, prices: pd.DataFrame, stock_info: pd.Series) -> tuple:
        """
        Apply learned entry filters
        """
        ticker = stock_info.get('ticker', 'UNKNOWN')

        if len(prices) < 20:
            return False, f"{ticker}: Insufficient price history"

        latest = prices.iloc[-1]

        # Use learned RSI thresholds
        rsi = latest['rsi_14']
        rsi_low = self.learned_params['rsi_threshold_low']
        rsi_high = self.learned_params['rsi_threshold_high']

        if pd.isna(rsi) or rsi < rsi_low or rsi > rsi_high:
            return False, f"{ticker}: RSI {rsi:.1f} outside learned range [{rsi_low:.0f}, {rsi_high:.0f}]"

        # Use learned z-score threshold
        z_score = latest['z_score']
        z_threshold = self.learned_params['z_score_threshold']

        if pd.isna(z_score) or z_score > z_threshold:
            return False, f"{ticker}: Z-score {z_score:.2f} above learned threshold {z_threshold:.2f}"

        # Days to ex-div must be in learned optimal days
        days_to_ex = stock_info['days_to_ex_div']
        if days_to_ex not in self.learned_params['optimal_entry_days']:
            return False, f"{ticker}: Days to ex-div {days_to_ex} not in learned window {self.learned_params['optimal_entry_days']}"

        # Volatility filter (still use config for safety)
        volatility = latest['volatility_20']
        if pd.isna(volatility) or volatility > entry_config.max_realized_vol:
            return False, f"{ticker}: Volatility {volatility*100:.1f}% too high"

        return True, "Passed all learned filters"

    def _calculate_learned_expected_return(self, price: float, stock_info: pd.Series,
                                          prices: pd.DataFrame) -> float:
        """
        Calculate expected return using learned parameters
        """
        if price <= 0:
            return 0

        dividend_amount = stock_info['amount']
        ticker = stock_info.get('ticker', '')
        sector = stock_info.get('sector', 'Unknown')

        # Use learned (or stock-specific) capture rate
        capture_rate = self._get_stock_capture_rate(ticker)

        # Base return from dividend capture
        dividend_return = (dividend_amount * capture_rate) / price

        # Sector adjustment
        sector_adj = 0
        if sector in self.learned_params['sector_performance']:
            sector_multiplier = self.learned_params['sector_performance'][sector]
            # Normalize sector multiplier (assume baseline is 2% return)
            sector_adj = (sector_multiplier - 0.02) * 0.5  # Scale down sector impact

        # Mean reversion component (still useful)
        latest = prices.iloc[-1]
        z_score = latest['z_score']

        # If oversold (negative z), add positive reversion expectation
        if not pd.isna(z_score) and z_score < 0:
            mr_return = abs(z_score) * 0.005  # 0.5% per std dev
        else:
            mr_return = 0

        # Total expected return
        total_return = dividend_return + sector_adj + mr_return

        return total_return

    def _get_stock_capture_rate(self, ticker: str) -> float:
        """Get stock-specific capture rate or default"""
        if ticker in self.learned_params['stock_specific_rates']:
            return self.learned_params['stock_specific_rates'][ticker]
        else:
            return self.learned_params['avg_dividend_capture_rate']

    def _calculate_stop_loss(self, entry_price: float, dividend_amount: float) -> float:
        """Calculate stop loss price"""
        if entry_price <= 0:
            return 0

        # Use learned parameters - tighter stop if we have high confidence
        hard_stop = entry_price * (1 - exit_config.hard_stop_pct)

        if exit_config.use_dividend_stop:
            div_stop = entry_price - dividend_amount
            return max(hard_stop, div_stop)

        return hard_stop

    def check_exit_signals(self, current_date: str) -> List[Dict]:
        """
        Check exit signals using learned optimal holding period
        """
        exit_signals = []
        current_dt = pd.to_datetime(current_date).tz_localize(None)

        for ticker, position in self.positions.items():
            # Get current price
            prices = self.dm.get_stock_prices(
                ticker,
                (current_dt - timedelta(days=30)).strftime('%Y-%m-%d'),
                current_date
            )

            if len(prices) == 0:
                continue

            current_price = prices['close'].iloc[-1]

            # Calculate P&L
            entry_price = position['entry_price']
            pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

            # Days held
            entry_dt = pd.to_datetime(position['entry_date']).tz_localize(None)
            days_held = (current_dt - entry_dt).days

            # Days since ex-dividend
            ex_div_dt = pd.to_datetime(position['ex_div_date']).tz_localize(None)
            days_since_ex_div = (current_dt - ex_div_dt).days

            exit_reason = None

            # 1. Stop loss
            if current_price <= position['stop_loss']:
                exit_reason = 'stop_loss'

            # 2. Profit target (learned or default)
            elif pnl_pct >= exit_config.profit_target_absolute:
                exit_reason = 'profit_target_absolute'

            # 3. Learned optimal hold period (exit after learned days post ex-div)
            elif current_dt >= ex_div_dt:
                if days_since_ex_div >= position.get('target_hold_days', self.learned_params['optimal_hold_days']):
                    exit_reason = 'learned_optimal_hold_period'

            # 4. Max holding (safety)
            elif days_held >= exit_config.max_holding_days:
                exit_reason = 'max_holding_period'

            # 5. Trailing stop
            if exit_config.trailing_stop_enabled and 'trailing_stop' in position:
                if current_price <= position['trailing_stop']:
                    exit_reason = 'trailing_stop'

            # Update trailing stop if in profit
            if exit_config.trailing_stop_enabled and pnl_pct >= exit_config.trailing_stop_activation:
                new_trailing_stop = current_price * (1 - exit_config.trailing_stop_distance)
                if 'trailing_stop' not in position or new_trailing_stop > position['trailing_stop']:
                    position['trailing_stop'] = new_trailing_stop

            # Generate exit signal
            if exit_reason:
                exit_signals.append({
                    'ticker': ticker,
                    'exit_date': current_date,
                    'exit_price': current_price,
                    'exit_reason': exit_reason,
                    'days_held': days_held,
                    'pnl_pct': pnl_pct,
                    **position
                })

        return exit_signals

    def open_position(self, signal: Dict) -> bool:
        """Open a new position"""
        ticker = signal['ticker']

        if ticker in self.positions:
            print(f"⚠️  Position already exists for {ticker}")
            return False

        self.positions[ticker] = signal

        print(f"📈 OPENED: {ticker} @ ${signal['entry_price']:.2f} "
              f"(Div: ${signal['dividend_amount']:.2f}, Expected Return: {signal['expected_return']*100:.2f}%)")

        return True

    def close_position(self, exit_signal: Dict) -> Dict:
        """Close an existing position"""
        ticker = exit_signal['ticker']

        if ticker not in self.positions:
            print(f"⚠️  No position found for {ticker}")
            return None

        # Calculate final P&L
        entry_price = exit_signal['entry_price']
        exit_price = exit_signal['exit_price']
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0
        pnl_amount = exit_price - entry_price

        closed_position = {
            **exit_signal,
            'pnl_pct': pnl_pct,
            'pnl_amount_per_share': pnl_amount,
        }

        self.position_history.append(closed_position)
        del self.positions[ticker]

        print(f"📉 CLOSED: {ticker} @ ${exit_price:.2f} "
              f"(P&L: {pnl_pct*100:+.2f}%, Reason: {exit_signal['exit_reason']})")

        return closed_position

    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio statistics"""
        num_positions = len(self.positions)

        if num_positions == 0:
            return {
                'num_positions': 0,
                'total_exposure': 0,
                'sectors': {},
                'avg_quality_score': 0,
                'avg_days_to_ex_div': 0
            }

        sectors = {}
        total_quality = 0
        total_days_to_ex_div = 0

        for ticker, pos in self.positions.items():
            sector = pos.get('sector', 'Unknown')
            sectors[sector] = sectors.get(sector, 0) + 1
            total_quality += pos['quality_score']
            total_days_to_ex_div += pos['days_to_ex_div']

        return {
            'num_positions': num_positions,
            'sectors': sectors,
            'avg_quality_score': total_quality / num_positions,
            'avg_days_to_ex_div': total_days_to_ex_div / num_positions
        }

    def get_trade_statistics(self) -> Dict:
        """Get statistics from closed trades"""
        if len(self.position_history) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'avg_pnl': 0,
                'profit_factor': 0,
                'avg_holding_days': 0
            }

        trades = pd.DataFrame(self.position_history)

        total_trades = len(trades)
        wins = trades[trades['pnl_pct'] > 0]
        losses = trades[trades['pnl_pct'] <= 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0
        avg_pnl = trades['pnl_pct'].mean()

        total_wins = wins['pnl_pct'].sum() if len(wins) > 0 else 0
        total_losses = abs(losses['pnl_pct'].sum()) if len(losses) > 0 else 1
        profit_factor = total_wins / total_losses if total_losses > 0 else 0

        avg_holding_days = trades['days_held'].mean()

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_pnl': avg_pnl,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding_days
        }
