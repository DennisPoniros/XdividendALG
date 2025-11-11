"""
Backtesting Engine for Dividend Capture Algorithm
Runs simulations with transaction costs and walk-forward validation
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
from strategy import DividendCaptureStrategy
from risk_manager import RiskManager


class Backtester:
    """
    Comprehensive backtesting engine with transaction costs and risk management
    """
    
    def __init__(self, start_date: str, end_date: str, initial_capital: float):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        
        # Initialize components
        self.dm = DataManager()
        self.strategy = DividendCaptureStrategy(self.dm)
        self.risk_manager = RiskManager(initial_capital)
        
        # Results tracking
        self.equity_curve = []
        self.trade_log = []
        self.daily_positions = []
        
        # Performance cache
        self.performance_metrics = {}
    
    def run_backtest(self, mode: str = 'full') -> Dict:
        """
        Run backtest simulation
        
        Args:
            mode: 'full' for entire period, 'walk_forward' for validation
            
        Returns:
            Dictionary of performance results
        """
        print("\n" + "="*80)
        print("🚀 STARTING BACKTEST")
        print("="*80)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print(f"Mode: {mode}")
        print("="*80 + "\n")
        
        if mode == 'walk_forward' and backtest_config.use_walk_forward:
            results = self._run_walk_forward()
        else:
            results = self._run_full_backtest()
        
        print("\n" + "="*80)
        print("✅ BACKTEST COMPLETED")
        print("="*80 + "\n")
        
        return results
    
    def _run_full_backtest(self) -> Dict:
        """Run backtest for entire period"""
        
        # Get dividend calendar for entire period
        print("📅 Loading dividend calendar...")
        dividend_calendar = self.dm.get_dividend_calendar(
            self.start_date, 
            self.end_date,
            lookback_days=30
        )
        
        if len(dividend_calendar) == 0:
            print("⚠️  No dividend events found. Cannot run backtest.")
            return {}
        
        # Generate trading dates
        trading_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
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
        
        # 1. Check exit signals for existing positions
        exit_signals = self.strategy.check_exit_signals(current_date)
        
        for exit_signal in exit_signals:
            self._execute_exit(exit_signal, current_date)
        
        # 2. Screen for new entry opportunities
        screened_stocks = self.dm.screen_stocks(dividend_calendar, current_date)
        
        if len(screened_stocks) > 0:
            # 3. Generate entry signals
            entry_signals = self.strategy.generate_entry_signals(screened_stocks, current_date)
            
            # 4. Risk management: allocate capital
            allocated_signals = self.risk_manager.get_position_allocation(
                entry_signals,
                self.strategy.positions
            )
            
            # 5. Execute entries
            for signal in allocated_signals:
                self._execute_entry(signal, current_date)
        
        # 6. Update portfolio value and risk metrics
        portfolio_value = self._calculate_portfolio_value(current_date)
        self.risk_manager.update_capital(portfolio_value)
        
        # 7. Record equity curve
        self.equity_curve.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': self.risk_manager.cash,
            'num_positions': len(self.strategy.positions)
        })
        
        # 8. Check circuit breakers
        if len(self.equity_curve) >= 2:
            returns = self._get_returns_series()
            should_stop, reason = self.risk_manager.should_stop_trading(returns)
            
            if should_stop:
                print(f"\n🛑 CIRCUIT BREAKER TRIGGERED: {reason}")
                # Close all positions
                for ticker in list(self.strategy.positions.keys()):
                    exit_signal = {
                        'ticker': ticker,
                        'exit_date': current_date,
                        'exit_price': self.strategy.positions[ticker]['entry_price'],
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
        
        # Commission (usually $0 now)
        commission = backtest_config.commission_per_trade
        
        # SEC fees
        sec_fee = position_value * backtest_config.sec_fee_per_dollar
        
        # Total cost
        total_cost = position_value + slippage_cost + commission + sec_fee
        
        # Check if we have enough cash
        if total_cost > self.risk_manager.cash:
            return False
        
        # Adjust entry price for slippage
        effective_entry_price = entry_price * (1 + backtest_config.slippage_bps / 10000)
        
        # Update signal with actual execution details
        signal['entry_price'] = effective_entry_price
        signal['shares'] = shares
        signal['position_value'] = shares * effective_entry_price
        signal['entry_costs'] = slippage_cost + commission + sec_fee
        
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
            'costs': slippage_cost + commission + sec_fee
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
        
        # Slippage (negative on exit - we get less)
        slippage_cost = position_value * (backtest_config.slippage_bps / 10000)
        
        # Commission
        commission = backtest_config.commission_per_trade
        
        # SEC fees
        sec_fee = position_value * backtest_config.sec_fee_per_dollar
        
        # Total proceeds (net of costs)
        total_proceeds = position_value - slippage_cost - commission - sec_fee
        
        # Adjust exit price for slippage
        effective_exit_price = exit_price * (1 - backtest_config.slippage_bps / 10000)
        
        # Update exit signal
        exit_signal['exit_price'] = effective_exit_price
        exit_signal['exit_costs'] = slippage_cost + commission + sec_fee
        
        # Close position
        closed_position = self.strategy.close_position(exit_signal)
        
        if closed_position:
            # Update cash
            self.risk_manager.update_cash(total_proceeds)
            
            # Log trade
            self.trade_log.append({
                'date': current_date,
                'action': 'SELL',
                'ticker': ticker,
                'shares': shares,
                'price': effective_exit_price,
                'value': position_value,
                'costs': slippage_cost + commission + sec_fee,
                'pnl': closed_position['pnl_amount_per_share'] * shares,
                'pnl_pct': closed_position['pnl_pct']
            })
        
        return True
    
    def _calculate_portfolio_value(self, current_date: str) -> float:
        """Calculate total portfolio value (cash + positions)"""
        
        cash = self.risk_manager.cash
        
        # Value of open positions
        position_value = 0
        
        for ticker, position in self.strategy.positions.items():
            # Get current price
            try:
                prices = self.dm.get_stock_prices(
                    ticker,
                    (pd.to_datetime(current_date) - timedelta(days=5)).strftime('%Y-%m-%d'),
                    current_date
                )
                
                if len(prices) > 0:
                    current_price = prices['close'].iloc[-1]
                    shares = position['shares']
                    position_value += shares * current_price
                else:
                    # Use entry price if no current price available
                    position_value += position['position_value']
            except:
                # Fallback to entry value
                position_value += position['position_value']
        
        total_value = cash + position_value
        
        return total_value
    
    def _get_returns_series(self) -> pd.Series:
        """Calculate daily returns from equity curve"""
        
        if len(self.equity_curve) < 2:
            return pd.Series()
        
        df = pd.DataFrame(self.equity_curve)
        df['returns'] = df['portfolio_value'].pct_change()
        
        return df['returns'].dropna()
    
    def _calculate_performance(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        
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
        
        # Risk-free rate (daily)
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
        
        # Win rate and trade statistics
        trade_stats = self.strategy.get_trade_statistics()
        
        # Value at Risk
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
            
            # DataFrames for plotting
            'equity_curve': equity_df,
            'returns': returns,
            'drawdown': drawdown,
        }
        
        # Print summary
        self._print_performance_summary()
        
        return self.performance_metrics
    
    def _print_performance_summary(self):
        """Print formatted performance summary"""
        
        m = self.performance_metrics
        
        print("\n" + "="*80)
        print("📈 PERFORMANCE SUMMARY")
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
        
        print("\n🎯 DAILY STATISTICS")
        print(f"  Avg Daily Return:    {m['avg_daily_return_pct']:>15.4f}%")
        print(f"  Best Day:            {m['best_day_pct']:>15.2f}%")
        print(f"  Worst Day:           {m['worst_day_pct']:>15.2f}%")
        print(f"  Positive Days:       {m['positive_day_rate']*100:>15.1f}%")
        
        print("\n" + "="*80)
        
        # Color-coded assessment
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
    
    def _run_walk_forward(self) -> Dict:
        """
        Run walk-forward analysis for robustness testing
        """
        print("\n🔄 Running Walk-Forward Analysis...")
        
        all_results = []
        
        start_date = pd.to_datetime(self.start_date)
        end_date = pd.to_datetime(self.end_date)
        
        train_days = backtest_config.walk_forward_train_days
        test_days = backtest_config.walk_forward_test_days
        step_days = backtest_config.walk_forward_step_days
        
        current_start = start_date
        
        while current_start + timedelta(days=train_days + test_days) <= end_date:
            
            train_end = current_start + timedelta(days=train_days)
            test_end = train_end + timedelta(days=test_days)
            
            print(f"\n📅 Window: Train {current_start.date()} to {train_end.date()}, "
                  f"Test {train_end.date()} to {test_end.date()}")
            
            # Run test period (we don't actually train parameters, but this simulates it)
            test_backtest = Backtester(
                train_end.strftime('%Y-%m-%d'),
                test_end.strftime('%Y-%m-%d'),
                self.initial_capital
            )
            
            results = test_backtest._run_full_backtest()
            
            if results:
                results['window_start'] = train_end.strftime('%Y-%m-%d')
                results['window_end'] = test_end.strftime('%Y-%m-%d')
                all_results.append(results)
            
            # Move window forward
            current_start += timedelta(days=step_days)
        
        # Aggregate walk-forward results
        if len(all_results) > 0:
            wf_summary = {
                'num_windows': len(all_results),
                'avg_sharpe': np.mean([r['sharpe_ratio'] for r in all_results]),
                'avg_annual_return': np.mean([r['annual_return'] for r in all_results]),
                'avg_max_drawdown': np.mean([r['max_drawdown'] for r in all_results]),
                'avg_win_rate': np.mean([r['win_rate'] for r in all_results]),
                'consistency': sum(1 for r in all_results if r['sharpe_ratio'] > 1.0) / len(all_results)
            }
            
            print("\n" + "="*80)
            print("🔄 WALK-FORWARD ANALYSIS RESULTS")
            print("="*80)
            print(f"  Number of Windows:     {wf_summary['num_windows']}")
            print(f"  Avg Sharpe Ratio:      {wf_summary['avg_sharpe']:.2f}")
            print(f"  Avg Annual Return:     {wf_summary['avg_annual_return']*100:.2f}%")
            print(f"  Avg Max Drawdown:      {wf_summary['avg_max_drawdown']*100:.2f}%")
            print(f"  Avg Win Rate:          {wf_summary['avg_win_rate']*100:.1f}%")
            print(f"  Consistency (Sharpe>1): {wf_summary['consistency']*100:.1f}%")
            print("="*80 + "\n")
            
            return wf_summary
        
        return {}
    
    def export_results(self, filepath: str = '/home/claude/dividend_algo/backtest_results.csv'):
        """Export trade log to CSV"""
        
        if len(self.trade_log) > 0:
            df = pd.DataFrame(self.trade_log)
            df.to_csv(filepath, index=False)
            print(f"✅ Results exported to {filepath}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_train_test_split():
    """Run backtest with separate train/test periods"""
    
    print("\n" + "="*80)
    print("🔬 RUNNING TRAIN/TEST SPLIT BACKTEST")
    print("="*80 + "\n")
    
    # Training period
    print("1️⃣  TRAINING PERIOD")
    print("-" * 80)
    train_bt = Backtester(
        data_config.train_start,
        data_config.train_end,
        backtest_config.initial_capital
    )
    train_results = train_bt.run_backtest(mode='full')
    
    # Testing period  
    print("\n2️⃣  TESTING PERIOD (OUT-OF-SAMPLE)")
    print("-" * 80)
    test_bt = Backtester(
        data_config.test_start,
        data_config.test_end,
        backtest_config.initial_capital
    )
    test_results = test_bt.run_backtest(mode='full')
    
    # Compare results
    print("\n" + "="*80)
    print("📊 TRAIN vs TEST COMPARISON")
    print("="*80)
    
    if train_results and test_results:
        comparison = pd.DataFrame({
            'Train': [
                f"{train_results['annual_return_pct']:.2f}%",
                f"{train_results['sharpe_ratio']:.2f}",
                f"{train_results['max_drawdown_pct']:.2f}%",
                f"{train_results['win_rate_pct']:.1f}%",
                train_results['total_trades']
            ],
            'Test': [
                f"{test_results['annual_return_pct']:.2f}%",
                f"{test_results['sharpe_ratio']:.2f}",
                f"{test_results['max_drawdown_pct']:.2f}%",
                f"{test_results['win_rate_pct']:.1f}%",
                test_results['total_trades']
            ]
        }, index=['Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate', 'Total Trades'])
        
        print(comparison)
        print("="*80 + "\n")
        
        # Check for overfitting
        sharpe_diff = abs(train_results['sharpe_ratio'] - test_results['sharpe_ratio'])
        if sharpe_diff < 0.5:
            print("✅ ROBUST: Similar performance between train/test (low overfitting)")
        else:
            print("⚠️  WARNING: Significant performance difference (possible overfitting)")
    
    return train_results, test_results


if __name__ == '__main__':
    # Run simple backtest
    bt = Backtester(
        '2023-01-01',
        '2024-10-31',
        100_000
    )
    results = bt.run_backtest(mode='full')
