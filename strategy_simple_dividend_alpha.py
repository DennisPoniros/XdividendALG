"""
Simple Dividend Alpha Strategy
===============================

SIMPLIFIED approach focused on execution and results, not complexity.

THESIS:
Buy quality dividend stocks when oversold, hold through dividend, exit on bounce.
Simple, proven, profitable.

ENTRY:
- 3-7 days before ex-dividend date
- RSI < 50 (not overbought)
- Price below 20-day MA (mean reversion opportunity)
- Dividend yield > 1.5%

POSITION SIZING:
- Fixed 4% per position
- Max 20 positions (80% deployed)
- Equal weight (simplicity)

EXIT:
- Profit target: +2.5%
- Stop loss: -1.5% (STRICT)
- Time stop: 12 days
- Ex-div exit: Sell 1 day after ex-div if up >0.5%

EXPECTED:
- 60-100 trades/year
- 60% win rate
- 1.8% avg win, -1.2% avg loss
- Sharpe > 1.2
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class SimpleDividendAlphaStrategy:
    """Simple dividend strategy optimized for consistent alpha."""

    def __init__(self, data_manager):
        self.dm = data_manager
        self.positions = {}

        self.params = {
            # Entry
            'days_before_exdiv_min': 3,
            'days_before_exdiv_max': 7,
            'max_rsi': 50,
            'min_div_yield': 0.015,  # 1.5%

            # Sizing
            'position_size': 0.04,  # 4%
            'max_positions': 20,

            # Exit
            'profit_target': 0.025,  # +2.5%
            'stop_loss': 0.015,  # -1.5%
            'max_holding_days': 12,
            'exdiv_exit_profit': 0.005,  # 0.5%
        }

    def screen_candidates(self, current_date: str) -> List[Dict]:
        """Screen for dividend opportunities."""
        current_dt = pd.to_datetime(current_date)
        candidates = []

        # Get dividend calendar
        start = (current_dt - timedelta(days=2)).strftime('%Y-%m-%d')
        end = (current_dt + timedelta(days=10)).strftime('%Y-%m-%d')

        try:
            div_calendar = self.dm.get_dividend_calendar(start, end)
        except:
            return []

        if len(div_calendar) == 0:
            return []

        for _, event in div_calendar.iterrows():
            try:
                ticker = event['ticker']
                ex_date_raw = event.get('ex_dividend_date', event.get('ex_date'))
                if pd.isna(ex_date_raw):
                    continue

                ex_div_date = pd.to_datetime(ex_date_raw)
                if pd.isna(ex_div_date):
                    continue

                # Remove timezone if present
                if hasattr(ex_div_date, 'tz_localize'):
                    ex_div_date = ex_div_date.tz_localize(None)
                elif hasattr(ex_div_date, 'tz'):
                    ex_div_date = ex_div_date.replace(tzinfo=None)

                dividend_amount = event.get('dividend_amount', event.get('amount', 0))

                days_to_exdiv = (ex_div_date - current_dt).days

                # Check if in entry window
                if not (self.params['days_before_exdiv_min'] <= days_to_exdiv <= self.params['days_before_exdiv_max']):
                    continue

                # Get price data - use longer lookback to ensure data availability
                try:
                    prices = self.dm.get_stock_prices(
                        ticker,
                        (current_dt - timedelta(days=40)).strftime('%Y-%m-%d'),
                        current_date
                    )
                except:
                    continue

                if len(prices) < 25:
                    continue

                # Get current price (last available)
                current_price = prices['close'].iloc[-1]

                if current_price <= 0:
                    continue

                # Calculate dividend yield
                div_yield = dividend_amount / current_price if current_price > 0 else 0
                if div_yield < self.params['min_div_yield']:
                    continue

                # Calculate simple indicators
                sma_20 = prices['close'].rolling(20).mean().iloc[-1]

                # Calculate RSI
                delta = prices['close'].diff()
                gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[-1]
                loss = -delta.where(delta < 0, 0).rolling(14).mean().iloc[-1]
                rs = gain / loss if loss > 0 else 100
                rsi = 100 - (100 / (1 + rs)) if rs != 100 else 100

                # Entry filters
                if rsi > self.params['max_rsi']:
                    continue

                # Mean reversion: price below MA
                if current_price > sma_20:
                    continue

                # Passed all filters
                candidates.append({
                    'ticker': ticker,
                    'price': current_price,
                    'dividend_amount': dividend_amount,
                    'div_yield': div_yield,
                    'ex_div_date': ex_div_date,
                    'days_to_exdiv': days_to_exdiv,
                    'rsi': rsi,
                    'sma_20': sma_20,
                })

            except Exception as e:
                continue

        return candidates

    def generate_entry_signals(self, current_date: str, capital: float, current_positions: Dict) -> List[Dict]:
        """Generate entry signals."""
        if len(current_positions) >= self.params['max_positions']:
            return []

        candidates = self.screen_candidates(current_date)

        # Sort by dividend yield (highest first)
        candidates = sorted(candidates, key=lambda x: x['div_yield'], reverse=True)

        signals = []
        for candidate in candidates:
            if len(current_positions) + len(signals) >= self.params['max_positions']:
                break

            signals.append(candidate)

        return signals

    def generate_exit_signals(self, current_date: str, current_positions: Dict) -> List[Dict]:
        """Generate exit signals."""
        exits = []
        current_dt = pd.to_datetime(current_date)

        for ticker, position in current_positions.items():
            try:
                # Get current price
                prices = self.dm.get_stock_prices(
                    ticker,
                    (current_dt - timedelta(days=5)).strftime('%Y-%m-%d'),
                    current_date
                )

                if len(prices) == 0:
                    continue

                current_price = prices['close'].iloc[-1]
                entry_price = position['entry_price']
                entry_date = pd.to_datetime(position['entry_date'])

                # Remove timezone from entry_date if present
                if hasattr(entry_date, 'tz_localize'):
                    entry_date = entry_date.tz_localize(None)
                elif hasattr(entry_date, 'tz'):
                    entry_date = entry_date.replace(tzinfo=None)

                days_held = (current_dt - entry_date).days
                pnl_pct = (current_price - entry_price) / entry_price

                # RULE 1: HARD STOP
                if pnl_pct <= -self.params['stop_loss']:
                    exits.append({'ticker': ticker, 'reason': 'stop_loss', 'pnl_pct': pnl_pct})
                    continue

                # RULE 2: Profit target
                if pnl_pct >= self.params['profit_target']:
                    exits.append({'ticker': ticker, 'reason': 'profit_target', 'pnl_pct': pnl_pct})
                    continue

                # RULE 3: Ex-div exit (1 day after ex-div, if profitable)
                ex_div_date = position.get('ex_div_date')
                if ex_div_date:
                    if hasattr(ex_div_date, 'tz_localize'):
                        ex_div_date = ex_div_date.tz_localize(None)
                    elif hasattr(ex_div_date, 'tz'):
                        ex_div_date = ex_div_date.replace(tzinfo=None)

                    days_since_exdiv = (current_dt - ex_div_date).days
                    if days_since_exdiv >= 1 and pnl_pct >= self.params['exdiv_exit_profit']:
                        exits.append({'ticker': ticker, 'reason': 'exdiv_exit', 'pnl_pct': pnl_pct})
                        continue

                # RULE 4: Time stop
                if days_held >= self.params['max_holding_days']:
                    exits.append({'ticker': ticker, 'reason': 'time_stop', 'pnl_pct': pnl_pct})
                    continue

            except:
                continue

        return exits

    def get_statistics(self) -> Dict:
        return {}
