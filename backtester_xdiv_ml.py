"""
Backtester for X-Dividend ML Strategy
Supports training period followed by testing period
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import (
    data_config, backtest_config, risk_config, analytics_config
)
from data_manager import DataManager
from strategy_xdiv_ml import XDividendMLStrategy
from risk_manager import RiskManager


class XDividendMLBacktester:
    """
    Backtester with training/testing split for ML strategy
    """

    def __init__(self, train_start: str, train_end: str,
                 test_start: str, test_end: str, initial_capital: float):
        self.train_start = train_start
        self.train_end = train_end
        self.test_start = test_start
        self.test_end = test_end
        self.initial_capital = initial_capital

        # Initialize components
        self.dm = DataManager()
        self.strategy = XDividendMLStrategy(self.dm, use_training=True)
        self.risk_manager = RiskManager(initial_capital)

        # Results tracking
        self.equity_curve = []
        self.trade_log = []
        self.daily_positions = []

        # Cost tracking
        self.total_entry_costs = 0
        self.total_exit_costs = 0
        self.total_slippage = 0
        self.total_commissions = 0
        self.total_sec_fees = 0

        # Performance cache
        self.performance_metrics = {}
        self.training_results = {}

    def run_backtest_with_training(self) -> Dict:
        """
        Run complete backtest with training followed by testing

        Returns:
            Dictionary with training and testing results
        """
        print("\n" + "="*80)
        print("🚀 X-DIVIDEND ML STRATEGY BACKTEST")
        print("="*80)
        print(f"Training Period: {self.train_start} to {self.train_end}")
        print(f"Testing Period:  {self.test_start} to {self.test_end}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print("="*80 + "\n")

        # Step 1: Train the strategy
        print("=" * 80)
        print("PHASE 1: TRAINING")
        print("=" * 80)
        self.strategy.train(self.train_start, self.train_end, verbose=True)
        self.training_results = self.strategy.training_metrics

        # Step 2: Test the strategy
        print("\n" + "=" * 80)
        print("PHASE 2: TESTING (OUT-OF-SAMPLE)")
        print("=" * 80)
        test_results = self._run_test_period()

        # Combine results
        combined_results = {
            'training': self.training_results,
            'testing': test_results,
            'equity_curve': pd.DataFrame(self.equity_curve),
            'trade_log': pd.DataFrame(self.trade_log),
        }

        return combined_results

    def _run_test_period(self) -> Dict:
        """Run backtest on test period using trained strategy"""

        # Get dividend calendar for test period
        print(f"\n📅 Loading dividend calendar for {self.test_start} to {self.test_end}...")
        dividend_calendar = self.dm.get_dividend_calendar(
            self.test_start,
            self.test_end,
            lookback_days=30
        )

        if len(dividend_calendar) == 0:
            print("⚠️  No dividend events found. Cannot run backtest.")
            return {}

        # Generate trading dates
        trading_dates = pd.date_range(
            start=self.test_start,
            end=self.test_end,
            freq='B'  # Business days
        )

        print(f"📊 Simulating {len(trading_dates)} trading days...\n")

        # Main simulation loop
        for date in trading_dates:
            current_date = date.strftime('%Y-%m-%d')
            self._simulate_trading_day(current_date, dividend_calendar)

            # Print progress every month
            if date.day == 1 or date == trading_dates[-1]:
                portfolio_value = self._calculate_portfolio_value(current_date)
                print(f"📈 {current_date}: Portfolio = ${portfolio_value:,.0f}, "
                      f"Positions = {len(self.strategy.positions)}, "
                      f"Trades = {len(self.strategy.position_history)}")

        # Calculate final results
        results = self._calculate_performance()

        return results

    def _simulate_trading_day(self, current_date: str, dividend_calendar: pd.DataFrame):
        """Simulate a single trading day"""

        # 1. Check exit signals
        exit_signals = self.strategy.check_exit_signals(current_date)

        for exit_signal in exit_signals:
            self._execute_exit(exit_signal, current_date)

        # 2. Screen for new entries
        screened_stocks = self.dm.screen_stocks(dividend_calendar, current_date)

        if len(screened_stocks) > 0:
            # 3. Generate entry signals (using learned parameters)
            entry_signals = self.strategy.generate_entry_signals(screened_stocks, current_date)

            # 4. Risk management
            allocated_signals = self.risk_manager.get_position_allocation(
                entry_signals,
                self.strategy.positions
            )

            # 5. Execute entries
            for signal in allocated_signals:
                self._execute_entry(signal, current_date)

        # 6. Update portfolio value
        portfolio_value = self._calculate_portfolio_value(current_date)
        self.risk_manager.update_capital(portfolio_value)

        # 7. Record equity curve
        self.equity_curve.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': self.risk_manager.cash,
            'num_positions': len(self.strategy.positions)
        })

        # 8. Check circuit breakers (if enabled)
        if risk_config.use_circuit_breakers and len(self.equity_curve) >= 2:
            returns = self._get_returns_series()
            should_stop, reason = self.risk_manager.should_stop_trading(returns)

            if should_stop:
                print(f"\n🛑 CIRCUIT BREAKER TRIGGERED: {reason}")
                # Close all positions
                for ticker in list(self.strategy.positions.keys()):
                    try:
                        prices = self.dm.get_stock_prices(
                            ticker,
                            (pd.to_datetime(current_date).tz_localize(None) - timedelta(days=5)).strftime('%Y-%m-%d'),
                            current_date
                        )
                        current_price = prices['close'].iloc[-1] if len(prices) > 0 else self.strategy.positions[ticker]['entry_price']
                    except:
                        current_price = self.strategy.positions[ticker]['entry_price']

                    exit_signal = {
                        'ticker': ticker,
                        'exit_date': current_date,
                        'exit_price': current_price,
                        'exit_reason': 'circuit_breaker',
                        **self.strategy.positions[ticker]
                    }
                    self._execute_exit(exit_signal, current_date)

    def _execute_entry(self, signal: Dict, current_date: str) -> bool:
        """Execute entry with transaction costs"""

        ticker = signal['ticker']
        shares = signal['shares']
        entry_price = signal['entry_price']

        # Calculate transaction costs
        position_value = shares * entry_price

        # Slippage
        slippage_cost = position_value * (backtest_config.slippage_bps / 10000)

        # Commission
        commission = backtest_config.commission_per_trade

        # SEC fees
        sec_fee = position_value * backtest_config.sec_fee_per_dollar

        # Total cost
        total_cost = position_value + slippage_cost + commission + sec_fee

        # Check cash
        if total_cost > self.risk_manager.cash:
            return False

        # Adjust entry price for slippage
        effective_entry_price = entry_price * (1 + backtest_config.slippage_bps / 10000)

        # Update signal
        signal['entry_price'] = effective_entry_price
        signal['shares'] = shares
        signal['position_value'] = shares * effective_entry_price
        signal['entry_costs'] = slippage_cost + commission + sec_fee

        # Track costs
        self.total_entry_costs += slippage_cost + commission + sec_fee
        self.total_slippage += slippage_cost
        self.total_commissions += commission
        self.total_sec_fees += sec_fee

        # Open position
        self.strategy.open_position(signal)

        # Update cash
        self.risk_manager.update_cash(-total_cost)

        # Log trade
        self.trade_log.append({
            'date': current_date,
            'action': 'BUY',
            'ticker': ticker,
            'shares': shares,
            'price': effective_entry_price,
            'value': position_value,
            'costs': slippage_cost + commission + sec_fee,
            'slippage': slippage_cost,
            'commission': commission,
            'sec_fee': sec_fee
        })

        return True

    def _execute_exit(self, exit_signal: Dict, current_date: str) -> bool:
        """Execute exit with transaction costs"""

        ticker = exit_signal['ticker']

        if ticker not in self.strategy.positions:
            return False

        position = self.strategy.positions[ticker]
        shares = position['shares']
        exit_price = exit_signal['exit_price']

        # Calculate transaction costs
        position_value = shares * exit_price

        # Slippage
        slippage_cost = position_value * (backtest_config.slippage_bps / 10000)

        # Commission
        commission = backtest_config.commission_per_trade

        # SEC fees
        sec_fee = position_value * backtest_config.sec_fee_per_dollar

        # Total proceeds
        total_proceeds = position_value - slippage_cost - commission - sec_fee

        # Adjust exit price
        effective_exit_price = exit_price * (1 - backtest_config.slippage_bps / 10000)

        # Update exit signal
        exit_signal['exit_price'] = effective_exit_price
        exit_signal['exit_costs'] = slippage_cost + commission + sec_fee

        # Track costs
        self.total_exit_costs += slippage_cost + commission + sec_fee
        self.total_slippage += slippage_cost
        self.total_commissions += commission
        self.total_sec_fees += sec_fee

        # Close position
        closed_position = self.strategy.close_position(exit_signal)

        if closed_position:
            # Update cash
            self.risk_manager.update_cash(total_proceeds)

            # Calculate P&L
            total_pnl = closed_position['pnl_amount_per_share'] * shares
            entry_costs = position.get('entry_costs', 0)
            exit_costs = slippage_cost + commission + sec_fee
            total_costs = entry_costs + exit_costs

            # Price movement P&L
            price_movement_pnl = (exit_price - position['entry_price']) * shares

            # Log trade
            self.trade_log.append({
                'date': current_date,
                'action': 'SELL',
                'ticker': ticker,
                'shares': shares,
                'price': effective_exit_price,
                'value': position_value,
                'costs': slippage_cost + commission + sec_fee,
                'slippage': slippage_cost,
                'commission': commission,
                'sec_fee': sec_fee,
                'pnl': total_pnl,
                'pnl_pct': closed_position['pnl_pct'],
                'price_movement_pnl': price_movement_pnl,
                'total_costs_paid': total_costs,
                'pnl_before_costs': price_movement_pnl,
                'exit_reason': exit_signal.get('exit_reason', 'unknown')
            })

        return True

    def _calculate_portfolio_value(self, current_date: str) -> float:
        """Calculate total portfolio value"""

        cash = self.risk_manager.cash
        position_value = 0

        for ticker, position in self.strategy.positions.items():
            try:
                prices = self.dm.get_stock_prices(
                    ticker,
                    (pd.to_datetime(current_date).tz_localize(None) - timedelta(days=5)).strftime('%Y-%m-%d'),
                    current_date
                )

                if len(prices) > 0:
                    current_price = prices['close'].iloc[-1]
                    shares = position['shares']
                    position_value += shares * current_price
                else:
                    position_value += position['position_value']
            except:
                position_value += position['position_value']

        return cash + position_value

    def _get_returns_series(self) -> pd.Series:
        """Calculate daily returns"""

        if len(self.equity_curve) < 2:
            return pd.Series()

        df = pd.DataFrame(self.equity_curve)
        df['returns'] = df['portfolio_value'].pct_change()

        return df['returns'].dropna()

    def _calculate_performance(self) -> Dict:
        """Calculate performance metrics"""

        if len(self.equity_curve) == 0:
            return {}

        print("\n📊 Calculating performance metrics...")

        # Convert to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df = equity_df.set_index('date')

        # Calculate returns
        equity_df['returns'] = equity_df['portfolio_value'].pct_change()
        returns = equity_df['returns'].dropna()

        # Basic metrics
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)

        # Risk-free rate
        rf_daily = (1 + analytics_config.risk_free_rate) ** (1/252) - 1

        # Sharpe Ratio
        excess_returns = returns - rf_daily
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Sortino Ratio
        downside_returns = returns[returns < rf_daily]
        downside_std = downside_returns.std()
        sortino = excess_returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Calmar Ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        trade_stats = self.strategy.get_trade_statistics()

        # VaR
        var_95 = self.risk_manager.calculate_portfolio_var(returns, 0.95)
        cvar_95 = self.risk_manager.calculate_cvar(returns, 0.95)

        # Compile results
        self.performance_metrics = {
            # Returns
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'annual_return': annual_return,
            'total_return_pct': total_return * 100,
            'annual_return_pct': annual_return * 100,

            # Risk
            'volatility_daily': daily_vol,
            'volatility_annual': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'calmar_ratio': calmar,

            # Risk metrics
            'var_95': var_95,
            'cvar_95': cvar_95,
            'var_95_pct': var_95 * 100,
            'cvar_95_pct': cvar_95 * 100,

            # Trade statistics
            'total_trades': trade_stats['total_trades'],
            'win_rate': trade_stats['win_rate'],
            'win_rate_pct': trade_stats['win_rate'] * 100,
            'avg_win': trade_stats['avg_win'],
            'avg_loss': trade_stats['avg_loss'],
            'avg_win_pct': trade_stats['avg_win'] * 100,
            'avg_loss_pct': trade_stats['avg_loss'] * 100,
            'profit_factor': trade_stats['profit_factor'],
            'avg_holding_days': trade_stats['avg_holding_days'],

            # Efficiency
            'avg_daily_return': returns.mean(),
            'avg_daily_return_pct': returns.mean() * 100,
            'best_day': returns.max(),
            'worst_day': returns.min(),
            'best_day_pct': returns.max() * 100,
            'worst_day_pct': returns.min() * 100,

            # Consistency
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            'positive_day_rate': (returns > 0).sum() / len(returns) if len(returns) > 0 else 0,

            # Monthly returns
            'monthly_returns': equity_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1),

            # Transaction costs
            'total_transaction_costs': self.total_entry_costs + self.total_exit_costs,
            'total_entry_costs': self.total_entry_costs,
            'total_exit_costs': self.total_exit_costs,
            'total_slippage': self.total_slippage,
            'total_commissions': self.total_commissions,
            'total_sec_fees': self.total_sec_fees,
            'costs_as_pct_of_capital': (self.total_entry_costs + self.total_exit_costs) / self.initial_capital * 100,
            'avg_cost_per_trade': (self.total_entry_costs + self.total_exit_costs) / (trade_stats['total_trades'] / 2) if trade_stats['total_trades'] > 0 else 0,

            # DataFrames
            'equity_curve': equity_df,
            'returns': returns,
            'drawdown': drawdown,
        }

        # Print summary
        self._print_performance_summary()

        return self.performance_metrics

    def _print_performance_summary(self):
        """Print performance summary"""

        m = self.performance_metrics

        print("\n" + "="*80)
        print("📈 TEST PERIOD PERFORMANCE SUMMARY")
        print("="*80)

        print("\n💰 RETURNS")
        print(f"  Initial Capital:     ${m['initial_capital']:>15,.0f}")
        print(f"  Final Value:         ${m['final_value']:>15,.0f}")
        print(f"  Total Return:        {m['total_return_pct']:>15.2f}%")
        print(f"  Annual Return:       {m['annual_return_pct']:>15.2f}%")

        print("\n⚠️  RISK METRICS")
        print(f"  Annual Volatility:   {m['volatility_annual']*100:>15.2f}%")
        print(f"  Sharpe Ratio:        {m['sharpe_ratio']:>15.2f}")
        print(f"  Sortino Ratio:       {m['sortino_ratio']:>15.2f}")
        print(f"  Calmar Ratio:        {m['calmar_ratio']:>15.2f}")
        print(f"  Max Drawdown:        {m['max_drawdown_pct']:>15.2f}%")
        print(f"  VaR 95%:             {m['var_95_pct']:>15.2f}%")
        print(f"  CVaR 95%:            {m['cvar_95_pct']:>15.2f}%")

        print("\n📊 TRADE STATISTICS")
        print(f"  Total Trades:        {m['total_trades']:>15}")
        print(f"  Win Rate:            {m['win_rate_pct']:>15.2f}%")
        print(f"  Avg Win:             {m['avg_win_pct']:>15.2f}%")
        print(f"  Avg Loss:            {m['avg_loss_pct']:>15.2f}%")
        print(f"  Profit Factor:       {m['profit_factor']:>15.2f}")
        print(f"  Avg Hold Period:     {m['avg_holding_days']:>15.1f} days")

        print("\n💸 TRANSACTION COSTS")
        print(f"  Total Costs:         ${m['total_transaction_costs']:>15,.2f}")
        print(f"  Costs as % Capital:  {m['costs_as_pct_of_capital']:>15.3f}%")
        print(f"  Avg Cost/Trade:      ${m['avg_cost_per_trade']:>15,.2f}")

        print("\n🎯 DAILY STATISTICS")
        print(f"  Avg Daily Return:    {m['avg_daily_return_pct']:>15.4f}%")
        print(f"  Best Day:            {m['best_day_pct']:>15.2f}%")
        print(f"  Worst Day:           {m['worst_day_pct']:>15.2f}%")
        print(f"  Positive Days:       {m['positive_day_rate']*100:>15.1f}%")

        print("\n" + "="*80)

        # Assessment vs targets
        if m['sharpe_ratio'] >= 1.5:
            print("✅ EXCELLENT: Sharpe ratio exceeds target (>1.5)")
        elif m['sharpe_ratio'] >= 1.0:
            print("✓  GOOD: Sharpe ratio above 1.0")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Sharpe ratio below 1.0")

        if m['win_rate_pct'] >= 55:
            print("✅ EXCELLENT: Win rate exceeds target (>55%)")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Win rate below 55%")

        if m['annual_return_pct'] >= 10:
            print("✅ EXCELLENT: Annual return exceeds target (>10%)")
        else:
            print("⚠️  NEEDS IMPROVEMENT: Annual return below 10%")

        print("="*80 + "\n")

    def export_results(self, filepath: str = '/mnt/user-data/outputs/xdiv_ml_results.csv'):
        """Export results to CSV"""

        if len(self.trade_log) > 0:
            import os
            save_dir = os.path.dirname(filepath)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

            df = pd.DataFrame(self.trade_log)
            df.to_csv(filepath, index=False)
            print(f"✅ Results exported to {filepath}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    # Run backtest with training
    bt = XDividendMLBacktester(
        train_start='2018-01-01',
        train_end='2022-12-31',
        test_start='2023-01-01',
        test_end='2024-10-31',
        initial_capital=100_000
    )

    results = bt.run_backtest_with_training()

    # Export results
    bt.export_results()

    print("\n✅ Backtest completed!")
