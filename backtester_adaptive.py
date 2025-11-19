"""
Backtester for Adaptive Dividend Mean Reversion Strategy
Ensures STRICT stop loss enforcement and accurate performance tracking
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from strategy_adaptive_div_mr import AdaptiveDividendMeanReversionStrategy
from mock_data_manager import MockDataManager


class AdaptiveBacktester:
    """
    Backtester with strict stop loss enforcement.
    """

    def __init__(self, start_date: str, end_date: str, initial_capital: float = 100_000,
                 use_mock_data: bool = True):
        """Initialize backtester."""
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.capital = initial_capital

        # Initialize data manager
        if use_mock_data:
            self.dm = MockDataManager()
        else:
            from data_manager import DataManager
            self.dm = DataManager()

        # Initialize strategy
        self.strategy = AdaptiveDividendMeanReversionStrategy(self.dm)

        # State tracking
        self.positions = {}
        self.trade_log = []
        self.equity_curve = []
        self.trade_count = 0

    def run_backtest(self) -> Dict:
        """Run backtest and return performance metrics."""
        print(f"\n{'='*80}")
        print("🚀 ADAPTIVE DIVIDEND MEAN REVERSION BACKTEST")
        print(f"{'='*80}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"{'='*80}\n")

        # Generate trading days
        date_range = pd.bdate_range(start=self.start_date, end=self.end_date)
        total_days = len(date_range)

        print(f"📊 Simulating {total_days} trading days...\n")

        for idx, current_date in enumerate(date_range):
            current_date_str = current_date.strftime('%Y-%m-%d')
            self._simulate_day(current_date_str)

            # Progress update every month
            if idx % 21 == 0 or idx == total_days - 1:
                pct_return = ((self.capital - self.initial_capital) / self.initial_capital) * 100
                print(f"📈 {current_date_str}: Value=${self.capital:,.0f} ({pct_return:+.2f}%), "
                      f"Positions={len(self.positions)}, Trades={self.trade_count}")

        # Calculate final metrics
        results = self._calculate_metrics()

        print(f"\n{'='*80}")
        print("📊 BACKTEST RESULTS")
        print(f"{'='*80}\n")

        print("💰 Returns:")
        print(f"   Total Return:        {results['total_return_pct']:>10.2f}%")
        print(f"   Annual Return:       {results['annual_return_pct']:>10.2f}%")
        print(f"   Final Value:         ${results['final_value']:>10,.0f}")

        print(f"\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio:        {results['sharpe_ratio']:>10.2f}")
        print(f"   Sortino Ratio:       {results['sortino_ratio']:>10.2f}")
        print(f"   Calmar Ratio:        {results['calmar_ratio']:>10.2f}")
        print(f"   Max Drawdown:        {results['max_drawdown_pct']:>10.2f}%")
        print(f"   Annual Volatility:   {results['annual_volatility_pct']:>10.2f}%")

        print(f"\n🎯 Trade Statistics:")
        print(f"   Total Trades:        {results['total_trades']:>10}")
        print(f"   Win Rate:            {results['win_rate_pct']:>10.2f}%")
        print(f"   Average Win:         {results['avg_win_pct']:>10.2f}%")
        print(f"   Average Loss:        {results['avg_loss_pct']:>10.2f}%")
        print(f"   Profit Factor:       {results['profit_factor']:>10.2f}")
        print(f"   Avg Holding Days:    {results['avg_holding_days']:>10.1f}")
        print(f"   Best Trade:          {results['best_trade_pct']:>10.2f}%")
        print(f"   Worst Trade:         {results['worst_trade_pct']:>10.2f}%")

        print(f"\n💸 Costs:")
        print(f"   Total Slippage:      ${results['total_slippage']:>10,.2f}")

        print(f"\n{'='*80}")

        # Performance assessment
        if results['sharpe_ratio'] >= 1.0:
            print("✅ SUCCESS: Sharpe ratio >= 1.0!")
        else:
            print("❌ NEEDS WORK: Sharpe ratio below target")

        if results['win_rate_pct'] < 55:
            print("⚠️  Win rate below target")

        print(f"{'='*80}\n")

        return results

    def _simulate_day(self, current_date: str):
        """Simulate one trading day with STRICT stop enforcement."""

        # 1. FIRST: Check stop losses on existing positions BEFORE anything else
        self._enforce_stops(current_date)

        # 2. Update position values
        self._update_positions(current_date)

        # 3. Check exit signals (profit targets, mean reversion, etc.)
        exit_signals = self.strategy.generate_exit_signals(current_date, self.positions)
        for exit_signal in exit_signals:
            self._execute_exit(current_date, exit_signal)

        # 4. Calculate current equity
        current_equity = self._calculate_equity(current_date)

        # 5. Generate entry signals for new positions
        entry_signals = self.strategy.generate_entry_signals(
            current_date,
            current_equity,
            self.positions
        )

        # 6. Execute entries
        for entry_signal in entry_signals:
            self._execute_entry(current_date, entry_signal)

        # 7. Record equity curve
        self.equity_curve.append({
            'date': current_date,
            'equity': current_equity,
            'positions': len(self.positions),
        })

    def _enforce_stops(self, current_date: str):
        """
        Enforce hard stop losses FIRST, before any other logic.
        This ensures stops actually work.
        """
        to_exit = []

        for ticker, position in self.positions.items():
            try:
                current_price = self.dm.get_current_price(ticker, current_date)
                if current_price is None:
                    continue

                entry_price = position['entry_price']
                pnl_pct = (current_price - entry_price) / entry_price

                # HARD STOP
                if pnl_pct <= -self.strategy.params['hard_stop_pct']:
                    to_exit.append({
                        'ticker': ticker,
                        'reason': 'stop_loss',
                        'pnl_pct': pnl_pct,
                        'price': current_price,
                    })
            except:
                continue

        # Execute stops immediately
        for exit_info in to_exit:
            ticker = exit_info['ticker']
            position = self.positions[ticker]

            entry_date = position['entry_date']
            days_held = (pd.to_datetime(current_date) - pd.to_datetime(entry_date)).days

            # Log trade
            self.trade_log.append({
                'entry_date': entry_date,
                'exit_date': current_date,
                'ticker': ticker,
                'entry_price': position['entry_price'],
                'exit_price': exit_info['price'],
                'shares': position['shares'],
                'pnl_pct': exit_info['pnl_pct'],
                'pnl_dollars': position['shares'] * (exit_info['price'] - position['entry_price']),
                'exit_reason': 'stop_loss',
                'days_held': days_held,
                'signal_type': position.get('signal_type', 'unknown'),
            })

            # Update capital
            exit_value = position['shares'] * exit_info['price']
            slippage = exit_value * 0.0005  # 5 bps
            self.capital += exit_value - slippage

            # Remove position
            del self.positions[ticker]
            self.trade_count += 1
            self.strategy.stopped_out_count += 1

    def _update_positions(self, current_date: str):
        """Update position values with current prices."""
        for ticker, position in self.positions.items():
            try:
                current_price = self.dm.get_current_price(ticker, current_date)
                if current_price:
                    position['current_price'] = current_price
                    position['current_value'] = position['shares'] * current_price
            except:
                pass

    def _execute_entry(self, current_date: str, signal: Dict):
        """Execute entry order."""
        ticker = signal['ticker']

        # Skip if already in position
        if ticker in self.positions:
            return

        current_equity = self._calculate_equity(current_date)
        position_value = current_equity * signal['position_size_pct']

        # Calculate shares
        price = signal['current_price']
        shares = int(position_value / price)

        if shares <= 0:
            return

        actual_cost = shares * price
        slippage = actual_cost * 0.0005  # 5 bps
        total_cost = actual_cost + slippage

        # Check if we have enough capital
        if total_cost > self.capital:
            return

        # Execute trade
        self.positions[ticker] = {
            'ticker': ticker,
            'entry_date': current_date,
            'entry_price': price,
            'shares': shares,
            'current_price': price,
            'current_value': actual_cost,
            'signal_type': signal['signal_type'],
            'z_score': signal['z_score'],
            'rsi': signal['rsi'],
            'atr': signal['atr'],
            'sector': signal.get('sector', 'Unknown'),
            'high_water_mark': price,
        }

        self.capital -= total_cost

    def _execute_exit(self, current_date: str, exit_signal: Dict):
        """Execute exit order."""
        ticker = exit_signal['ticker']

        if ticker not in self.positions:
            return

        position = self.positions[ticker]
        current_price = self.dm.get_current_price(ticker, current_date)

        if current_price is None:
            return

        entry_date = position['entry_date']
        days_held = (pd.to_datetime(current_date) - pd.to_datetime(entry_date)).days

        # Calculate P&L
        pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        pnl_dollars = position['shares'] * (current_price - position['entry_price'])

        # Log trade
        self.trade_log.append({
            'entry_date': entry_date,
            'exit_date': current_date,
            'ticker': ticker,
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'shares': position['shares'],
            'pnl_pct': pnl_pct,
            'pnl_dollars': pnl_dollars,
            'exit_reason': exit_signal['reason'],
            'days_held': days_held,
            'signal_type': position.get('signal_type', 'unknown'),
        })

        # Update capital
        exit_value = position['shares'] * current_price
        slippage = exit_value * 0.0005  # 5 bps
        self.capital += exit_value - slippage

        # Remove position
        del self.positions[ticker]
        self.trade_count += 1

    def _calculate_equity(self, current_date: str) -> float:
        """Calculate total equity."""
        self._update_positions(current_date)
        position_value = sum(p.get('current_value', 0) for p in self.positions.values())
        return self.capital + position_value

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trade_log) if self.trade_log else pd.DataFrame()

        # Basic returns
        final_value = equity_df['equity'].iloc[-1] if len(equity_df) > 0 else self.initial_capital
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Annualized return
        days = (pd.to_datetime(self.end_date) - pd.to_datetime(self.start_date)).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Daily returns
        equity_df['daily_return'] = equity_df['equity'].pct_change()
        daily_returns = equity_df['daily_return'].dropna()

        # Volatility
        annual_vol = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0

        # Sharpe ratio (assume 4% risk-free rate)
        risk_free_daily = 0.04 / 252
        excess_returns = daily_returns - risk_free_daily
        sharpe = (excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if len(excess_returns) > 1 and excess_returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = excess_returns[excess_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 1 else annual_vol / np.sqrt(252)
        sortino = (excess_returns.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0

        # Max drawdown
        equity_df['cummax'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = abs(equity_df['drawdown'].min()) if len(equity_df) > 0 else 0

        # Calmar ratio
        calmar = annual_return / max_drawdown if max_drawdown > 0 else 0

        # Trade statistics
        if len(trades_df) > 0:
            wins = trades_df[trades_df['pnl_pct'] > 0]
            losses = trades_df[trades_df['pnl_pct'] <= 0]

            win_rate = len(wins) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0

            gross_profit = wins['pnl_dollars'].sum() if len(wins) > 0 else 0
            gross_loss = abs(losses['pnl_dollars'].sum()) if len(losses) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            avg_holding = trades_df['days_held'].mean()
            best_trade = trades_df['pnl_pct'].max()
            worst_trade = trades_df['pnl_pct'].min()
            total_slippage = len(trades_df) * 2 * (self.initial_capital * 0.02) * 0.0005
        else:
            win_rate = avg_win = avg_loss = 0
            profit_factor = avg_holding = 0
            best_trade = worst_trade = 0
            total_slippage = 0

        return {
            'final_value': final_value,
            'total_return_pct': total_return * 100,
            'annual_return_pct': annual_return * 100,
            'annual_volatility_pct': annual_vol * 100,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'max_drawdown_pct': max_drawdown * 100,
            'total_trades': len(trades_df),
            'win_rate_pct': win_rate * 100,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding,
            'best_trade_pct': best_trade * 100,
            'worst_trade_pct': worst_trade * 100,
            'total_slippage': total_slippage,
        }
