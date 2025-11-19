"""
Fixed X-Dividend ML Strategy - Removes Problematic Stop Losses

Critical Fix: Don't exit on dividend drop (it's expected!)
"""

# Import everything from the original strategy
from strategy_xdiv_ml import *


class XDividendMLStrategyFixed(XDividendMLStrategy):
    """
    Fixed version of X-Dividend ML Strategy

    Key Changes:
    1. NO stop losses (dividend drop is expected)
    2. Exit based on time after ex-div
    3. Optional profit target only
    """

    def _calculate_stop_loss(self, entry_price: float, dividend_amount: float) -> float:
        """
        FIXED: Return very low stop loss (only for disasters)

        Original problem: stop_loss = entry_price - dividend_amount
        This would trigger immediately after ex-div!

        New: Only exit on catastrophic losses (-15%)
        """
        if entry_price <= 0:
            return 0

        # Emergency stop only (catastrophic losses)
        emergency_stop = entry_price * 0.85  # -15%

        return emergency_stop

    def check_exit_signals(self, current_date: str) -> List[Dict]:
        """
        FIXED: Check exit signals without problematic stop losses

        Exit Rules:
        1. After ex-div + learned hold period (primary)
        2. Profit target (5%)
        3. Max holding (15 days)
        4. Emergency only (<-15%)
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

            # 1. EMERGENCY STOP ONLY (very wide, -15%)
            if pnl_pct <= -0.15:
                exit_reason = 'emergency_stop'

            # 2. Profit target (5%)
            elif pnl_pct >= 0.05:
                exit_reason = 'profit_target'

            # 3. Time-based exits (PRIMARY)
            elif days_held >= 15:  # Max holding
                exit_reason = 'max_holding_period'

            # 4. Post-dividend logic (learned hold period)
            elif current_dt >= ex_div_dt and days_since_ex_div >= position.get('target_hold_days', 7):
                exit_reason = 'learned_hold_period'

            # 5. RSI overbought (optional)
            elif len(prices) > 0:
                prices_with_ind = self.dm.calculate_technical_indicators(prices)
                rsi = prices_with_ind['rsi_14'].iloc[-1]
                if not pd.isna(rsi) and rsi > 75:  # Very overbought
                    exit_reason = 'rsi_overbought'

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
