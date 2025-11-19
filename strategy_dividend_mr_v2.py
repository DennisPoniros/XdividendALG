"""
Dividend Mean Reversion Strategy V2
More aggressive version designed to trade frequently and capture alpha

KEY CHANGES FROM V1:
1. **Relaxed Entry Filters**: Accept more signals (-0.5 z-score threshold)
2. **Longer Entry Windows**: Enter 1-7 days before OR 0-3 days after ex-div
3. **Dynamic Exits**: Exit when mean reversion complete OR profit target hit
4. **Higher Trade Frequency**: Target 50-100+ trades per year
5. **Sector Diversification**: Ensure broad exposure

PHILOSOPHY:
- Trade MORE frequently with SMALLER position sizes
- Focus on statistical edge from mean reversion
- Accept moderate win rate (55-60%) with good risk/reward
- Compound small gains consistently
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class DividendMeanReversionV2:
    """
    Aggressive dividend mean reversion strategy for higher Sharpe ratio
    """

    def __init__(self, data_manager):
        """Initialize strategy."""
        self.dm = data_manager
        self.positions = {}
        self.position_history = []

        # V2 Parameters - TUNED FOR HIGH TRADE FREQUENCY
        self.params = {
            # Entry windows (RELAXED)
            'pre_div_entry_window': (-7, -1),    # 7 days before ex-div
            'post_div_entry_window': (0, 3),     # Up to 3 days after

            # Entry thresholds (LOOSER)
            'min_zscore_entry': -0.5,  # Enter when z < -0.5 (mildly oversold)
            'min_rsi_entry': 25,        # RSI > 25 (not too extreme)
            'max_rsi_entry': 75,        # RSI < 75

            # Position sizing (SMALLER BUT MORE POSITIONS)
            'base_position_size': 0.015,  # 1.5% per position
            'max_position_size': 0.025,   # 2.5% max

            # Exit rules (ADAPTIVE)
            'profit_target_pct': 0.02,    # 2% profit target
            'stop_loss_pct': 0.015,       # 1.5% stop loss
            'max_holding_days': 8,        # Exit after 8 days
            'exit_on_zscore_positive': True,  # Exit when z-score > 0.5

            # Risk management
            'max_positions': 25,          # More positions = more diversification
            'max_sector_exposure': 0.35,  # Allow more sector concentration
            'min_cash_reserve': 0.15,     # Keep 15% cash

            # Quality filters (MINIMAL)
            'min_price': 5,                # No penny stocks
            'max_volatility': 0.60,       # Allow higher vol stocks
            'min_volume': 100_000,         # 100k shares/day minimum
        }

    def screen_candidates(self, current_date: str) -> List[Dict]:
        """
        Screen for trading opportunities - AGGRESSIVE VERSION

        Args:
            current_date: Current date

        Returns:
            List of candidates
        """
        current_dt = pd.to_datetime(current_date).tz_localize(None)
        candidates = []

        # Get wide dividend calendar window
        lookback = (current_dt - timedelta(days=7)).strftime('%Y-%m-%d')
        lookahead = (current_dt + timedelta(days=10)).strftime('%Y-%m-%d')

        div_calendar = self.dm.get_dividend_calendar(lookback, lookahead)

        if len(div_calendar) == 0:
            return []

        for _, event in div_calendar.iterrows():
            ticker = event['ticker']
            ex_div_date = pd.to_datetime(event['ex_dividend_date']).tz_localize(None)
            dividend_amount = event['dividend_amount']

            days_to_ex_div = (ex_div_date - current_dt).days

            # Check if in entry window
            pre_min, pre_max = self.params['pre_div_entry_window']
            post_min, post_max = self.params['post_div_entry_window']

            if pre_min <= days_to_ex_div <= pre_max:
                signal_type = 'pre_dividend'
            elif post_min <= days_to_ex_div <= post_max:
                signal_type = 'post_dividend'
            else:
                continue

            # Get price data
            price_start = (current_dt - timedelta(days=60)).strftime('%Y-%m-%d')
            prices = self.dm.get_stock_prices(ticker, price_start, current_date)

            if len(prices) < 20:
                continue

            # Calculate technicals
            prices = self.dm.calculate_technical_indicators(prices)
            latest = prices.iloc[-1]

            current_price = latest['close']
            z_score = latest['z_score']
            rsi = latest['rsi_14']
            volatility = latest['volatility_20']

            # MINIMAL filtering
            if current_price < self.params['min_price']:
                continue

            if pd.isna(z_score) or pd.isna(rsi) or pd.isna(volatility):
                continue

            # Z-score filter (RELAXED) - just need to be below mean
            if z_score >= self.params['min_zscore_entry']:
                continue

            # RSI filter (WIDE RANGE)
            if rsi < self.params['min_rsi_entry'] or rsi > self.params['max_rsi_entry']:
                continue

            # Volatility cap
            if volatility > self.params['max_volatility']:
                continue

            # Calculate expected return
            expected_return = self._estimate_return(
                z_score, volatility, dividend_amount, current_price, signal_type
            )

            # Position size based on conviction
            conviction = self._calculate_conviction(z_score, rsi, volatility)
            position_size = self.params['base_position_size'] * conviction
            position_size = min(position_size, self.params['max_position_size'])

            candidates.append({
                'ticker': ticker,
                'signal_type': signal_type,
                'ex_div_date': ex_div_date.strftime('%Y-%m-%d'),
                'days_to_ex_div': days_to_ex_div,
                'dividend_amount': dividend_amount,
                'dividend_yield': dividend_amount / current_price,
                'current_price': current_price,
                'z_score': z_score,
                'rsi': rsi,
                'volatility': volatility,
                'expected_return': expected_return,
                'conviction': conviction,
                'position_size': position_size,
                'sector': event.get('sector', 'Unknown'),
                'entry_date': current_date,
            })

        # Sort by expected return
        candidates.sort(key=lambda x: x['expected_return'], reverse=True)

        return candidates

    def _estimate_return(
        self,
        z_score: float,
        volatility: float,
        dividend: float,
        price: float,
        signal_type: str
    ) -> float:
        """Estimate expected return from trade."""

        # Mean reversion component: further from mean = higher expected return
        mr_return = abs(z_score) * volatility * 0.3

        # Dividend component
        if signal_type == 'pre_dividend':
            div_return = (dividend / price) * 0.25  # 25% capture efficiency
        else:
            div_return = (dividend / price) * 0.5   # 50% reversion potential

        return mr_return + div_return

    def _calculate_conviction(self, z_score: float, rsi: float, volatility: float) -> float:
        """Calculate conviction level 0-1."""

        # Z-score conviction: more extreme = higher
        z_conviction = min(1.0, abs(z_score) / 2.0)

        # RSI conviction: closer to oversold/overbought extremes
        if z_score < 0:  # Bearish z-score, want low RSI
            rsi_conviction = max(0, (50 - rsi) / 25)  # Higher when RSI < 50
        else:
            rsi_conviction = max(0, (rsi - 50) / 25)

        # Volatility: prefer moderate vol
        vol_conviction = max(0, 1.0 - volatility / 0.5)

        # Combine
        conviction = (z_conviction * 0.5 + rsi_conviction * 0.3 + vol_conviction * 0.2)

        return max(0.5, min(1.5, conviction))  # Clamp between 0.5 and 1.5

    def generate_entry_signals(
        self,
        current_date: str,
        current_capital: float,
        current_positions: Dict
    ) -> List[Dict]:
        """Generate entry signals."""

        # Don't add more if at max
        if len(current_positions) >= self.params['max_positions']:
            return []

        # Screen
        candidates = self.screen_candidates(current_date)

        if not candidates:
            return []

        # Track sector exposure
        sector_exposure = {}
        for pos in current_positions.values():
            sector = pos.get('sector', 'Unknown')
            sector_exposure[sector] = sector_exposure.get(sector, 0) + pos.get('position_size', 0.015)

        # Available cash
        reserved = current_capital * self.params['min_cash_reserve']
        available = current_capital - reserved

        signals = []

        for candidate in candidates:
            ticker = candidate['ticker']
            sector = candidate['sector']

            # Skip if have position
            if ticker in current_positions:
                continue

            # Sector limit
            current_sector_exp = sector_exposure.get(sector, 0)
            if current_sector_exp + candidate['position_size'] > self.params['max_sector_exposure']:
                continue

            # Calculate shares
            position_value = current_capital * candidate['position_size']

            if position_value > available:
                break

            shares = int(position_value / candidate['current_price'])

            if shares == 0:
                continue

            # Create signal
            signal = {
                'ticker': ticker,
                'action': 'BUY',
                'shares': shares,
                'entry_price': candidate['current_price'],
                'entry_date': current_date,
                'signal_type': candidate['signal_type'],
                'ex_div_date': candidate['ex_div_date'],
                'days_to_ex_div': candidate['days_to_ex_div'],
                'dividend_amount': candidate['dividend_amount'],
                'z_score': candidate['z_score'],
                'rsi': candidate['rsi'],
                'volatility': candidate['volatility'],
                'expected_return': candidate['expected_return'],
                'conviction': candidate['conviction'],
                'position_size': candidate['position_size'],
                'sector': sector,

                # Exit levels
                'profit_target': candidate['current_price'] * (1 + self.params['profit_target_pct']),
                'stop_loss': candidate['current_price'] * (1 - self.params['stop_loss_pct']),
            }

            signals.append(signal)
            available -= position_value
            sector_exposure[sector] = current_sector_exp + candidate['position_size']

            # Add multiple per day
            if len(signals) >= 8:
                break

        return signals

    def check_exit_signals(
        self,
        current_date: str,
        current_positions: Dict
    ) -> List[Dict]:
        """Check exits - ADAPTIVE."""

        exit_signals = []
        current_dt = pd.to_datetime(current_date).tz_localize(None)

        for ticker, position in current_positions.items():
            # Get current data
            price_start = (current_dt - timedelta(days=5)).strftime('%Y-%m-%d')
            prices = self.dm.get_stock_prices(ticker, price_start, current_date)

            if len(prices) == 0:
                continue

            prices = self.dm.calculate_technical_indicators(prices)
            current_price = prices['close'].iloc[-1]
            current_z = prices['z_score'].iloc[-1]

            # Calculate metrics
            entry_price = position['entry_price']
            entry_date = pd.to_datetime(position['entry_date']).tz_localize(None)
            days_held = (current_dt - entry_date).days

            pnl_pct = (current_price - entry_price) / entry_price if entry_price > 0 else 0

            exit_reason = None

            # Exit 1: Profit target
            if current_price >= position.get('profit_target', float('inf')):
                exit_reason = 'profit_target'

            # Exit 2: Stop loss
            elif current_price <= position.get('stop_loss', 0):
                exit_reason = 'stop_loss'

            # Exit 3: Mean reversion complete (z-score positive)
            elif self.params['exit_on_zscore_positive'] and not pd.isna(current_z):
                if current_z > 0.5:  # Above mean
                    exit_reason = 'mean_reversion_complete'

            # Exit 4: Time stop
            elif days_held >= self.params['max_holding_days']:
                exit_reason = 'time_stop'

            # Exit 5: Take profit on any gain after 5 days
            elif days_held >= 5 and pnl_pct > 0.01:  # 1% gain after 5 days
                exit_reason = 'take_profit_time'

            if exit_reason:
                exit_signals.append({
                    'ticker': ticker,
                    'action': 'SELL',
                    'shares': position['shares'],
                    'exit_price': current_price,
                    'exit_date': current_date,
                    'exit_reason': exit_reason,
                    'entry_price': entry_price,
                    'entry_date': position['entry_date'],
                    'days_held': days_held,
                    'pnl_pct': pnl_pct * 100,
                })

        return exit_signals

    def open_position(self, signal: Dict) -> bool:
        """Open position."""
        ticker = signal['ticker']

        if ticker in self.positions:
            return False

        self.positions[ticker] = signal

        print(f"📈 OPEN: {ticker} @ ${signal['entry_price']:.2f} "
              f"(Z={signal['z_score']:.2f}, {signal['signal_type']})")

        return True

    def close_position(self, exit_signal: Dict) -> Dict:
        """Close position."""
        ticker = exit_signal['ticker']

        if ticker not in self.positions:
            return None

        entry_price = exit_signal['entry_price']
        exit_price = exit_signal['exit_price']
        pnl_pct = (exit_price - entry_price) / entry_price if entry_price > 0 else 0

        closed = {**exit_signal, 'pnl_pct': pnl_pct}
        self.position_history.append(closed)

        del self.positions[ticker]

        print(f"📉 CLOSE: {ticker} @ ${exit_price:.2f} "
              f"(P&L: {pnl_pct*100:+.2f}%, {exit_signal['exit_reason']})")

        return closed

    def get_statistics(self) -> Dict:
        """Get statistics."""
        if not self.position_history:
            return {'total_trades': 0, 'win_rate': 0}

        df = pd.DataFrame(self.position_history)

        return {
            'total_trades': len(df),
            'win_rate': len(df[df['pnl_pct'] > 0]) / len(df),
            'avg_return': df['pnl_pct'].mean(),
            'avg_days_held': df['days_held'].mean(),
        }


if __name__ == '__main__':
    print("Dividend Mean Reversion V2 - Aggressive High-Frequency Version")
    print("Target: 50-100+ trades/year, Sharpe > 1.0")
