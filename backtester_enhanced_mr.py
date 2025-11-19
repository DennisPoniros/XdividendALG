"""
Backtester for Enhanced Dividend Mean Reversion Strategy
Simple, efficient backtesting focused on iterative improvement
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from data_manager import DataManager
from strategy_enhanced_mr import EnhancedDividendMeanReversionStrategy


class EnhancedMRBacktester:
    """
    Streamlined backtester for rapid iteration and optimization
    """

    def __init__(self, start_date: str, end_date: str, initial_capital: float):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.cash = initial_capital

        # Initialize components
        self.dm = DataManager()
        self.strategy = EnhancedDividendMeanReversionStrategy(self.dm)

        # Results tracking
        self.equity_curve = []
        self.trade_log = []
        self.daily_returns = []

        # Cost tracking
        self.slippage_bps = 5  # 0.05% per trade
        self.total_costs = 0

    def run_backtest(self) -> Dict:
        """
        Run backtest over specified period

        Returns:
            Dictionary with performance metrics
        """
        print("\n" + "="*80)
        print("🚀 ENHANCED DIVIDEND MEAN REVERSION BACKTEST")
        print("="*80)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Initial Capital: ${self.initial_capital:,.0f}")
        print("="*80 + "\n")

        # Generate trading dates
        trading_dates = pd.date_range(
            start=self.start_date,
            end=self.end_date,
            freq='B'  # Business days only
        )

        print(f"📊 Simulating {len(trading_dates)} trading days...\n")

        # Main simulation loop
        for i, date in enumerate(trading_dates):
            current_date = date.strftime('%Y-%m-%d')
            self._simulate_day(current_date)

            # Progress updates
            if i % 20 == 0 or i == len(trading_dates) - 1:
                portfolio_value = self._get_portfolio_value(current_date)
                total_return = ((portfolio_value - self.initial_capital) / self.initial_capital) * 100

                print(f"📈 {current_date}: Value=${portfolio_value:,.0f} "
                      f"({total_return:+.2f}%), Positions={len(self.strategy.positions)}, "
                      f"Trades={len(self.strategy.position_history)}")

        # Calculate performance
        results = self._calculate_performance()

        return results

    def _simulate_day(self, current_date: str):
        """Simulate one trading day"""

        # 1. Check exits first
        exit_signals = self.strategy.check_exit_signals(
            current_date,
            self.strategy.positions
        )

        for exit_signal in exit_signals:
            self._execute_exit(exit_signal)

        # 2. Generate entry signals
        portfolio_value = self._get_portfolio_value(current_date)
        entry_signals = self.strategy.generate_entry_signals(
            current_date,
            portfolio_value,
            self.strategy.positions
        )

        # 3. Execute entries
        for entry_signal in entry_signals:
            self._execute_entry(entry_signal, portfolio_value)

        # 4. Record daily state
        portfolio_value = self._get_portfolio_value(current_date)

        self.equity_curve.append({
            'date': current_date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'num_positions': len(self.strategy.positions),
        })

        # Calculate daily return
        if len(self.equity_curve) > 1:
            prev_value = self.equity_curve[-2]['portfolio_value']
            daily_return = (portfolio_value - prev_value) / prev_value
            self.daily_returns.append(daily_return)

    def _execute_entry(self, signal: Dict, portfolio_value: float) -> bool:
        """Execute buy order"""

        ticker = signal['ticker']
        shares = signal['shares']
        price = signal['entry_price']

        # Calculate costs
        trade_value = shares * price
        slippage = trade_value * (self.slippage_bps / 10000)
        total_cost = trade_value + slippage

        # Check if we have enough cash
        if total_cost > self.cash:
            return False

        # Execute trade
        self.cash -= total_cost
        self.total_costs += slippage

        # Open position in strategy
        self.strategy.open_position(signal)

        # Log trade
        self.trade_log.append({
            'date': signal['entry_date'],
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'value': trade_value,
            'cost': slippage,
            'signal_type': signal.get('signal_type', 'unknown'),
            'z_score': signal.get('z_score', 0),
            'conviction': signal.get('conviction', 0),
        })

        return True

    def _execute_exit(self, signal: Dict) -> bool:
        """Execute sell order"""

        ticker = signal['ticker']
        shares = signal['shares']
        price = signal['exit_price']

        # Calculate proceeds
        trade_value = shares * price
        slippage = trade_value * (self.slippage_bps / 10000)
        proceeds = trade_value - slippage

        # Execute trade
        self.cash += proceeds
        self.total_costs += slippage

        # Close position in strategy
        closed = self.strategy.close_position(signal)

        # Log trade
        if closed:
            self.trade_log.append({
                'date': signal['exit_date'],
                'ticker': ticker,
                'action': 'SELL',
                'shares': shares,
                'price': price,
                'value': trade_value,
                'cost': slippage,
                'exit_reason': signal.get('exit_reason', 'unknown'),
                'pnl_pct': signal.get('pnl_pct', 0),
                'days_held': signal.get('days_held', 0),
            })

        return True

    def _get_portfolio_value(self, current_date: str) -> float:
        """Calculate total portfolio value"""

        # Start with cash
        total_value = self.cash

        # Add value of open positions
        for ticker, position in self.strategy.positions.items():
            try:
                # Get current price
                prices = self.dm.get_stock_prices(
                    ticker,
                    (pd.to_datetime(current_date).tz_localize(None) - timedelta(days=3)).strftime('%Y-%m-%d'),
                    current_date
                )

                if len(prices) > 0:
                    current_price = prices['close'].iloc[-1]
                    shares = position['shares']
                    total_value += shares * current_price
                else:
                    # Use entry price if no data available
                    total_value += position['shares'] * position['entry_price']

            except:
                # Fallback to entry price
                total_value += position['shares'] * position['entry_price']

        return total_value

    def _calculate_performance(self) -> Dict:
        """Calculate comprehensive performance metrics"""

        if len(self.equity_curve) == 0:
            return {}

        equity_df = pd.DataFrame(self.equity_curve)
        equity_df['date'] = pd.to_datetime(equity_df['date'])
        equity_df.set_index('date', inplace=True)

        # Basic metrics
        final_value = equity_df['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        # Time period
        days = len(equity_df)
        years = days / 252

        # Annualized return
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # Daily returns
        returns = equity_df['portfolio_value'].pct_change().dropna()

        # Sharpe ratio (assuming 4% risk-free rate)
        risk_free_daily = 0.04 / 252
        excess_returns = returns - risk_free_daily
        sharpe_ratio = (excess_returns.mean() / excess_returns.std()) * np.sqrt(252) if excess_returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0

        # Maximum drawdown
        cummax = equity_df['portfolio_value'].cummax()
        drawdown = (equity_df['portfolio_value'] - cummax) / cummax
        max_drawdown = drawdown.min()

        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # Trade statistics
        if len(self.strategy.position_history) > 0:
            trades_df = pd.DataFrame(self.strategy.position_history)

            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl_pct'] > 0])
            win_rate = winning_trades / total_trades if total_trades > 0 else 0

            avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].mean() if (total_trades - winning_trades) > 0 else 0

            # Profit factor
            gross_profit = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].sum() if winning_trades > 0 else 0
            gross_loss = abs(trades_df[trades_df['pnl_pct'] <= 0]['pnl_pct'].sum()) if (total_trades - winning_trades) > 0 else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

            avg_holding_days = trades_df['days_held'].mean()

            # Best and worst trades
            best_trade = trades_df['pnl_pct'].max()
            worst_trade = trades_df['pnl_pct'].min()

        else:
            total_trades = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding_days = 0
            best_trade = 0
            worst_trade = 0

        # Compile results
        results = {
            # Returns
            'total_return_pct': total_return * 100,
            'annual_return_pct': annual_return * 100,
            'final_value': final_value,

            # Risk metrics
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'volatility_annual': returns.std() * np.sqrt(252) * 100,

            # Trade statistics
            'total_trades': total_trades,
            'win_rate_pct': win_rate * 100,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'profit_factor': profit_factor,
            'avg_holding_days': avg_holding_days,
            'best_trade_pct': best_trade * 100,
            'worst_trade_pct': worst_trade * 100,

            # Costs
            'total_costs': self.total_costs,

            # Period
            'trading_days': days,
            'years': years,
        }

        # Print summary
        self._print_results(results)

        # Store for later access
        self.performance_metrics = results

        return results

    def _print_results(self, results: Dict):
        """Print formatted results"""

        print("\n" + "="*80)
        print("📊 BACKTEST RESULTS")
        print("="*80)

        print(f"\n💰 Returns:")
        print(f"   Total Return:        {results['total_return_pct']:>10.2f}%")
        print(f"   Annual Return:       {results['annual_return_pct']:>10.2f}%")
        print(f"   Final Value:         ${results['final_value']:>10,.0f}")

        print(f"\n📈 Risk Metrics:")
        print(f"   Sharpe Ratio:        {results['sharpe_ratio']:>10.2f}")
        print(f"   Sortino Ratio:       {results['sortino_ratio']:>10.2f}")
        print(f"   Calmar Ratio:        {results['calmar_ratio']:>10.2f}")
        print(f"   Max Drawdown:        {results['max_drawdown_pct']:>10.2f}%")
        print(f"   Annual Volatility:   {results['volatility_annual']:>10.2f}%")

        print(f"\n🎯 Trade Statistics:")
        print(f"   Total Trades:        {results['total_trades']:>10.0f}")
        print(f"   Win Rate:            {results['win_rate_pct']:>10.2f}%")
        print(f"   Average Win:         {results['avg_win_pct']:>10.2f}%")
        print(f"   Average Loss:        {results['avg_loss_pct']:>10.2f}%")
        print(f"   Profit Factor:       {results['profit_factor']:>10.2f}")
        print(f"   Avg Holding Days:    {results['avg_holding_days']:>10.1f}")
        print(f"   Best Trade:          {results['best_trade_pct']:>10.2f}%")
        print(f"   Worst Trade:         {results['worst_trade_pct']:>10.2f}%")

        print(f"\n💸 Costs:")
        print(f"   Total Slippage:      ${results['total_costs']:>10,.2f}")

        print("\n" + "="*80)

        # Assessment
        if results['sharpe_ratio'] >= 1.0:
            print("✅ EXCELLENT: Sharpe ratio >= 1.0 achieved!")
        elif results['sharpe_ratio'] >= 0.7:
            print("⚠️  GOOD: Sharpe ratio decent but needs improvement")
        else:
            print("❌ NEEDS WORK: Sharpe ratio below target")

        if results['win_rate_pct'] >= 55:
            print("✅ EXCELLENT: Win rate above 55%")
        else:
            print("⚠️  Win rate below target")

        print("="*80 + "\n")


if __name__ == '__main__':
    # Quick test
    print("Enhanced Mean Reversion Backtester")
    print("Ready to run backtests for rapid iteration")
