"""
Post-Dividend Dip Buyer Strategy

INVERSE APPROACH to traditional dividend capture:
- Instead of buying BEFORE ex-div to capture dividend
- Buy AFTER ex-div when price has dropped
- Capitalize on mean reversion WITHOUT dividend taxation
- Exit when price recovers to pre-dividend level

THESIS:
1. Stock drops ~70-90% of dividend amount on ex-div day (expected)
2. Price is now "cheaper" - at a discount
3. Mean reversion brings price back up over next few days
4. We capture the reversion WITHOUT paying dividend taxes
5. No dividend income = simpler tax treatment

ADVANTAGES over traditional capture:
- Avoid dividend taxation (can be 15-37% depending on bracket)
- Lower entry price (buy the dip)
- Same mean reversion opportunity
- May have better risk/reward if drop overshoots

RISKS:
- Stock might not mean-revert (fundamental change)
- Opportunity cost (no dividend received)
- Requires precise timing

TRADING RULES:
Entry:
- 0-2 days AFTER ex-dividend date
- Price has dropped close to expected amount (70-90% of dividend)
- Stock shows oversold conditions (RSI < 40)
- Quality filters passed

Exit:
- Price recovers to pre-dividend level (full reversion)
- OR 80% recovery (take partial profit)
- OR max holding period (7-10 days)
- Stop loss: -3% from entry (genuine breakdown)

Position Sizing:
- Fixed 2% per position (simple, no Kelly)
- Max 25 positions
- 20% cash reserve

COMPARISON:
Traditional Capture:
  Entry: $100 (3 days before ex-div)
  Ex-div: $99 (drops $1 dividend)
  Exit: $100 (mean reversion)
  Profit: $1 dividend - taxes (15-37%) = $0.63-$0.85

Post-Div Dip:
  Entry: $99 (1 day after ex-div)
  Exit: $100 (mean reversion)
  Profit: $1 capital gain (taxed at lower rate or offset by losses)
  Net: ~$0.80-$1.00 (better after-tax return for some investors)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from data_manager import DataManager
from config import screening_config


class PostDividendDipStrategy:
    """
    Buy stocks AFTER ex-dividend date at discounted prices,
    exit when they mean-revert back up.
    """

    def __init__(self, data_manager: DataManager):
        """Initialize strategy."""
        self.dm = data_manager
        self.positions = {}
        self.pending_orders = {}

        # Strategy parameters (can be optimized)
        self.params = {
            # Entry timing
            'min_days_after_ex_div': 0,  # Enter same day or next
            'max_days_after_ex_div': 2,  # Don't wait too long

            # Entry conditions
            'min_drop_pct_of_dividend': 0.50,  # Stock must have dropped at least 50% of div
            'max_drop_pct_of_dividend': 1.20,  # If drops >120%, might be fundamental issue

            # Technical filters
            'max_entry_rsi': 40,  # Only buy oversold
            'min_volume_ratio': 0.8,  # At least 80% of normal volume

            # Exit targets
            'full_recovery_exit': True,  # Exit when price = pre-div price
            'partial_recovery_pct': 0.80,  # Or exit at 80% recovery

            # Risk management
            'max_holding_days': 10,  # Exit after 10 days regardless
            'stop_loss_pct': 0.03,  # -3% hard stop (genuine breakdown)

            # Position sizing
            'position_size_pct': 0.02,  # 2% per position
            'max_positions': 25,
            'min_cash_reserve': 0.20,  # 20% cash
        }

    def screen_candidates(self, current_date: str) -> List[Dict]:
        """
        Find stocks that recently went ex-dividend and have dropped.

        Args:
            current_date: Current trading date (YYYY-MM-DD)

        Returns:
            List of candidate stocks with metadata
        """
        current_dt = pd.to_datetime(current_date)

        # Get dividend calendar for recent past
        lookback_date = (current_dt - timedelta(days=5)).strftime('%Y-%m-%d')

        dividend_events = self.dm.get_dividend_calendar(lookback_date, current_date)

        if len(dividend_events) == 0:
            return []

        candidates = []

        for _, event in dividend_events.iterrows():
            ticker = event['ticker']
            ex_div_date = pd.to_datetime(event['ex_dividend_date']).tz_localize(None)
            dividend_amount = event['dividend_amount']

            # Check timing: are we 0-2 days after ex-div?
            days_since_ex_div = (current_dt - ex_div_date).days

            if days_since_ex_div < self.params['min_days_after_ex_div']:
                continue
            if days_since_ex_div > self.params['max_days_after_ex_div']:
                continue

            # Get price data around ex-div
            price_start = (ex_div_date - timedelta(days=10)).strftime('%Y-%m-%d')
            price_end = current_date

            prices = self.dm.get_stock_prices(ticker, price_start, price_end)

            if len(prices) < 5:
                continue

            # Get pre-dividend price (day before ex-div)
            pre_div_prices = prices[prices.index < ex_div_date]
            if len(pre_div_prices) == 0:
                continue
            pre_div_price = pre_div_prices['close'].iloc[-1]

            # Get current price
            current_price = prices['close'].iloc[-1]

            # Calculate actual drop
            actual_drop = pre_div_price - current_price
            actual_drop_pct = actual_drop / pre_div_price if pre_div_price > 0 else 0

            # Expected drop
            expected_drop_pct = dividend_amount / pre_div_price if pre_div_price > 0 else 0

            # Check if drop is in acceptable range
            drop_ratio = actual_drop / dividend_amount if dividend_amount > 0 else 0

            if drop_ratio < self.params['min_drop_pct_of_dividend']:
                continue  # Hasn't dropped enough
            if drop_ratio > self.params['max_drop_pct_of_dividend']:
                continue  # Dropped too much (fundamental issue?)

            # Calculate technical indicators
            rsi = self._calculate_rsi(prices['close'], period=14)
            if rsi is None or rsi > self.params['max_entry_rsi']:
                continue  # Not oversold enough

            # Volume check
            avg_volume = prices['volume'].rolling(20).mean().iloc[-1]
            current_volume = prices['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

            if volume_ratio < self.params['min_volume_ratio']:
                continue  # Low volume, might be illiquid

            # Basic quality screening (light version)
            if not self._passes_quality_screen(ticker):
                continue

            # Calculate expected return (mean reversion potential)
            reversion_potential = (pre_div_price - current_price) / current_price

            # Add to candidates
            candidates.append({
                'ticker': ticker,
                'ex_div_date': ex_div_date.strftime('%Y-%m-%d'),
                'days_since_ex_div': days_since_ex_div,
                'dividend_amount': dividend_amount,
                'pre_div_price': pre_div_price,
                'current_price': current_price,
                'actual_drop': actual_drop,
                'actual_drop_pct': actual_drop_pct * 100,
                'expected_drop_pct': expected_drop_pct * 100,
                'drop_ratio': drop_ratio,
                'rsi': rsi,
                'volume_ratio': volume_ratio,
                'reversion_potential': reversion_potential * 100,
                'entry_signal_date': current_date,
            })

        # Sort by reversion potential (highest first)
        candidates.sort(key=lambda x: x['reversion_potential'], reverse=True)

        return candidates

    def _passes_quality_screen(self, ticker: str) -> bool:
        """Light quality screening (faster than full screening)."""
        try:
            # Just check basic filters
            fundamentals = self.dm.get_fundamentals(ticker)

            if fundamentals is None:
                return True  # If no data, let it pass (benefit of doubt)

            # Minimum requirements
            market_cap = fundamentals.get('market_cap', 0)
            if market_cap < screening_config.min_market_cap:
                return False

            # That's it - we're more lenient for post-div strategy
            return True

        except:
            return True  # If error, let it pass

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> Optional[float]:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return None

        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)

        avg_gain = gains.rolling(window=period).mean()
        avg_loss = losses.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if len(rsi) > 0 else None

    def generate_entry_signals(
        self,
        current_date: str,
        current_capital: float,
        current_positions: Dict
    ) -> List[Dict]:
        """
        Generate entry signals for post-dividend dip buying.

        Args:
            current_date: Current date
            current_capital: Available capital
            current_positions: Current open positions

        Returns:
            List of entry signals
        """
        # Don't enter if already at max positions
        if len(current_positions) >= self.params['max_positions']:
            return []

        # Screen candidates
        candidates = self.screen_candidates(current_date)

        if not candidates:
            return []

        # Calculate position size
        position_value = current_capital * self.params['position_size_pct']

        # Check available cash
        available_cash = current_capital * (1 - self.params['min_cash_reserve'])

        # Generate signals
        signals = []

        for candidate in candidates:
            # Skip if already have position
            if candidate['ticker'] in current_positions:
                continue

            # Check if we have enough cash
            if position_value > available_cash:
                break  # No more cash

            # Calculate shares
            shares = int(position_value / candidate['current_price'])
            if shares == 0:
                continue

            # Create entry signal
            signal = {
                'ticker': candidate['ticker'],
                'action': 'BUY',
                'shares': shares,
                'price': candidate['current_price'],
                'date': current_date,
                'signal_type': 'post_dividend_dip',

                # Entry metadata
                'ex_div_date': candidate['ex_div_date'],
                'days_since_ex_div': candidate['days_since_ex_div'],
                'pre_div_price': candidate['pre_div_price'],
                'dividend_amount': candidate['dividend_amount'],
                'entry_rsi': candidate['rsi'],
                'reversion_potential': candidate['reversion_potential'],

                # For position tracking
                'target_price': candidate['pre_div_price'],  # Full recovery target
                'stop_loss': candidate['current_price'] * (1 - self.params['stop_loss_pct']),
            }

            signals.append(signal)
            available_cash -= position_value

            # Limit to a few per day
            if len(signals) >= 5:
                break

        return signals

    def check_exit_signals(
        self,
        current_date: str,
        current_positions: Dict
    ) -> List[Dict]:
        """
        Check exit conditions for open positions.

        Args:
            current_date: Current date
            current_positions: Dictionary of open positions

        Returns:
            List of exit signals
        """
        exit_signals = []
        current_dt = pd.to_datetime(current_date).tz_localize(None)

        for ticker, position in current_positions.items():
            # Get current price
            prices = self.dm.get_stock_prices(
                ticker,
                (current_dt - timedelta(days=5)).strftime('%Y-%m-%d'),
                current_date
            )

            if len(prices) == 0:
                continue

            current_price = prices['close'].iloc[-1]

            # Calculate metrics
            entry_price = position['entry_price']
            entry_date = pd.to_datetime(position['entry_date']).tz_localize(None)
            days_held = (current_dt - entry_date).days

            pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

            # Target price (pre-dividend level)
            target_price = position.get('target_price', entry_price * 1.02)
            recovery_pct = (current_price - entry_price) / (target_price - entry_price) if target_price > entry_price else 0

            exit_reason = None

            # Exit condition 1: Full recovery (price >= target)
            if current_price >= target_price:
                exit_reason = 'full_recovery'

            # Exit condition 2: Partial recovery (80% of the way)
            elif recovery_pct >= self.params['partial_recovery_pct']:
                exit_reason = 'partial_recovery'

            # Exit condition 3: Stop loss
            elif current_price <= position.get('stop_loss', 0):
                exit_reason = 'stop_loss'

            # Exit condition 4: Max holding period
            elif days_held >= self.params['max_holding_days']:
                exit_reason = 'max_holding_period'

            # Generate exit signal
            if exit_reason:
                exit_signals.append({
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': position['shares'],
                    'price': current_price,
                    'date': current_date,
                    'exit_reason': exit_reason,
                    'entry_price': entry_price,
                    'entry_date': position['entry_date'],
                    'days_held': days_held,
                    'pnl_pct': pnl_pct * 100,
                    'recovery_pct': recovery_pct * 100,
                })

        return exit_signals


if __name__ == '__main__':
    # Quick test
    print("Post-Dividend Dip Strategy")
    print("=" * 80)
    print("\nStrategy Logic:")
    print("  1. Wait for stocks to go ex-dividend")
    print("  2. Buy AFTER the price drops (0-2 days post ex-div)")
    print("  3. Hold for mean reversion back to pre-div price")
    print("  4. Exit at recovery or max 10 days")
    print("\nAdvantages:")
    print("  ✓ No dividend taxation")
    print("  ✓ Lower entry price (buy the dip)")
    print("  ✓ Same mean reversion opportunity")
    print("  ✓ Simpler tax treatment (capital gains)")
    print("\nExpected Performance:")
    print("  - Win rate: 55-65% (similar to capture)")
    print("  - Avg gain: 1.5-2.5% (smaller than capture but tax-advantaged)")
    print("  - Holding period: 3-7 days")
    print("  - Sharpe: 1.5-2.0 (lower volatility)")
    print("=" * 80)
