"""
Backtester for Post-Dividend Dip Strategy

Simpler than ML backtester - no training period needed.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from data_manager import DataManager
from strategy_post_div_dip import PostDividendDipStrategy


class PostDivDipBacktester:
    """Backtester for post-dividend dip buying strategy."""

    def __init__(
        self,
        start_date: str,
        end_date: str,
        initial_capital: float = 100_000
    ):
        """
        Initialize backtester.

        Args:
            start_date: Backtest start date (YYYY-MM-DD)
            end_date: Backtest end date (YYYY-MM-DD)
            initial_capital: Starting capital
        """
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital

        # Initialize components
        self.dm = DataManager()
        self.strategy = PostDividendDipStrategy(self.dm)

        # State
        self.cash = initial_capital
        self.positions = {}
        self.equity_curve = []
        self.trade_log = []

        # Costs (same as other strategies)
        self.slippage_bps = 5  # 0.05%
        self.total_costs = 0

    def run_backtest(self) -> Dict:
        """
        Run backtest over specified period.

        Returns:
            Dictionary with results
        """
        print("\n" + "="*80)
        print("🔄 POST-DIVIDEND DIP STRATEGY BACKTEST")
        print("="*80)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print("="*80 + "\n")

        # Get trading days
        trading_days = pd.bdate_range(
            start=self.start_date,
            end=self.end_date,
            freq='B'  # Business days
        )

        print(f"📅 Trading days: {len(trading_days)}")
        print(f"🎯 Strategy: Buy AFTER ex-div, sell at recovery\n")

        # Run backtest day by day
        for i, current_date in enumerate(trading_days):
            current_date_str = current_date.strftime('%Y-%m-%d')

            # Progress indicator
            if i % 50 == 0:
                progress = (i / len(trading_days)) * 100
                print(f"Progress: {progress:.1f}% - {current_date_str} - "
                      f"Positions: {len(self.positions)} - "
                      f"Cash: ${self.cash:,.0f}")

            # Check exits first
            exit_signals = self.strategy.check_exit_signals(
                current_date_str,
                self.positions
            )

            for exit_signal in exit_signals:
                self._execute_exit(exit_signal)

            # Update equity
            equity = self._calculate_equity(current_date_str)
            self.equity_curve.append({
                'date': current_date_str,
                'equity': equity,
                'cash': self.cash,
                'positions': len(self.positions)
            })

            # Generate entry signals
            entry_signals = self.strategy.generate_entry_signals(
                current_date_str,
                self.cash,
                self.positions
            )

            for entry_signal in entry_signals:
                self._execute_entry(entry_signal)

        # Close any remaining positions at end
        print("\n📊 Closing remaining positions...")
        self._close_all_positions(self.end_date)

        # Calculate metrics
        results = self._calculate_metrics()

        self._print_summary(results)

        return results

    def _execute_entry(self, signal: Dict):
        """Execute an entry order."""
        ticker = signal['ticker']
        shares = signal['shares']
        price = signal['price']
        date = signal['date']

        # Calculate costs
        notional = shares * price
        slippage = notional * (self.slippage_bps / 10000)
        total_cost = notional + slippage

        # Check if we have enough cash
        if total_cost > self.cash:
            return  # Skip if insufficient cash

        # Update cash
        self.cash -= total_cost
        self.total_costs += slippage

        # Create position
        self.positions[ticker] = {
            'shares': shares,
            'entry_price': price,
            'entry_date': date,
            'entry_cost': total_cost,
            'target_price': signal.get('target_price', price * 1.02),
            'stop_loss': signal.get('stop_loss', price * 0.97),
            'ex_div_date': signal.get('ex_div_date'),
            'pre_div_price': signal.get('pre_div_price'),
        }

        # Log entry
        self.trade_log.append({
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'slippage': slippage,
            'notional': notional,
        })

    def _execute_exit(self, signal: Dict):
        """Execute an exit order."""
        ticker = signal['ticker']

        if ticker not in self.positions:
            return

        position = self.positions[ticker]
        shares = signal['shares']
        price = signal['price']
        date = signal['date']

        # Calculate P&L
        notional = shares * price
        slippage = notional * (self.slippage_bps / 10000)
        proceeds = notional - slippage

        entry_cost = position['entry_cost']
        pnl = proceeds - entry_cost
        pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0

        # Update cash
        self.cash += proceeds
        self.total_costs += slippage

        # Log exit
        self.trade_log.append({
            'date': date,
            'ticker': ticker,
            'action': 'EXIT',
            'shares': shares,
            'price': price,
            'slippage': slippage,
            'notional': notional,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'exit_reason': signal.get('exit_reason'),
            'entry_date': position['entry_date'],
            'entry_price': position['entry_price'],
            'days_held': signal.get('days_held', 0),
        })

        # Remove position
        del self.positions[ticker]

    def _calculate_equity(self, current_date: str) -> float:
        """Calculate total equity (cash + position values)."""
        position_value = 0

        for ticker, position in self.positions.items():
            # Get current price
            try:
                prices = self.dm.get_stock_prices(
                    ticker,
                    (pd.to_datetime(current_date) - timedelta(days=5)).strftime('%Y-%m-%d'),
                    current_date
                )
                if len(prices) > 0:
                    current_price = prices['close'].iloc[-1]
                    position_value += position['shares'] * current_price
            except:
                # If can't get price, use entry price
                position_value += position['shares'] * position['entry_price']

        return self.cash + position_value

    def _close_all_positions(self, date: str):
        """Close all remaining positions at market."""
        for ticker in list(self.positions.keys()):
            position = self.positions[ticker]

            # Get final price
            try:
                prices = self.dm.get_stock_prices(
                    ticker,
                    (pd.to_datetime(date) - timedelta(days=5)).strftime('%Y-%m-%d'),
                    date
                )
                if len(prices) > 0:
                    exit_price = prices['close'].iloc[-1]
                else:
                    exit_price = position['entry_price']
            except:
                exit_price = position['entry_price']

            # Create exit signal
            entry_date = pd.to_datetime(position['entry_date'])
            current_date = pd.to_datetime(date)
            days_held = (current_date - entry_date).days

            exit_signal = {
                'ticker': ticker,
                'shares': position['shares'],
                'price': exit_price,
                'date': date,
                'exit_reason': 'backtest_end',
                'days_held': days_held,
            }

            self._execute_exit(exit_signal)

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        # Equity curve analysis
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)

        # Returns
        equity_df['returns'] = equity_df['equity'].pct_change()

        final_equity = equity_df['equity'].iloc[-1]
        total_return_pct = ((final_equity - self.initial_capital) / self.initial_capital) * 100

        # Annualized metrics
        days = len(equity_df)
        years = days / 252
        annual_return_pct = ((final_equity / self.initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0

        # Risk metrics
        daily_returns = equity_df['returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe = (annual_return_pct / 100) / volatility if volatility > 0 else 0

        # Drawdown
        cumulative = (1 + daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown_pct = drawdown.min() * 100

        # Trade statistics
        trades_df = pd.DataFrame(self.trade_log)
        exit_trades = trades_df[trades_df['action'] == 'EXIT']

        if len(exit_trades) > 0:
            total_trades = len(exit_trades)
            wins = exit_trades[exit_trades['pnl'] > 0]
            losses = exit_trades[exit_trades['pnl'] <= 0]

            win_rate_pct = (len(wins) / total_trades) * 100
            avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
            avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0
            profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else 0

            avg_hold = exit_trades['days_held'].mean()
            best_trade = exit_trades['pnl_pct'].max()
            worst_trade = exit_trades['pnl_pct'].min()
        else:
            total_trades = win_rate_pct = avg_win = avg_loss = profit_factor = 0
            avg_hold = best_trade = worst_trade = 0

        return {
            'final_equity': final_equity,
            'total_return_pct': total_return_pct,
            'annual_return_pct': annual_return_pct,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown_pct,
            'total_trades': total_trades,
            'win_rate_pct': win_rate_pct,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_hold_days': avg_hold,
            'best_trade_pct': best_trade,
            'worst_trade_pct': worst_trade,
            'total_costs': self.total_costs,
        }

    def _print_summary(self, results: Dict):
        """Print results summary."""
        print("\n" + "="*80)
        print("📈 BACKTEST RESULTS")
        print("="*80 + "\n")

        print("💰 RETURNS")
        print(f"  Initial Capital:     ${self.initial_capital:>12,.0f}")
        print(f"  Final Value:         ${results['final_equity']:>12,.0f}")
        print(f"  Total Return:        {results['total_return_pct']:>12.2f}%")
        print(f"  Annual Return:       {results['annual_return_pct']:>12.2f}%")

        print(f"\n📊 RISK METRICS")
        print(f"  Volatility:          {results['volatility']:>12.2f}%")
        print(f"  Sharpe Ratio:        {results['sharpe_ratio']:>12.2f}")
        print(f"  Max Drawdown:        {results['max_drawdown_pct']:>12.2f}%")

        print(f"\n📈 TRADE STATISTICS")
        print(f"  Total Trades:        {results['total_trades']:>12.0f}")
        print(f"  Win Rate:            {results['win_rate_pct']:>12.1f}%")
        print(f"  Avg Win:             ${results['avg_win']:>12.2f}")
        print(f"  Avg Loss:            ${results['avg_loss']:>12.2f}")
        print(f"  Profit Factor:       {results['profit_factor']:>12.2f}")
        print(f"  Avg Hold Period:     {results['avg_hold_days']:>12.1f} days")

        print(f"\n💸 COSTS")
        print(f"  Total Costs:         ${results['total_costs']:>12.2f}")

        print("\n" + "="*80)
