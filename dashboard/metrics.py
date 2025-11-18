"""
Advanced metrics calculator for backtesting analytics.

Calculates:
- Returns metrics (total, annualized, daily)
- Risk-adjusted metrics (Sharpe, Sortino, Omega, Calmar)
- Drawdown metrics (max drawdown, duration, recovery)
- Trade statistics (win rate, profit factor, expectancy)
- Benchmark comparisons (S&P 500)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta

# Optional import for benchmark data
try:
    import yfinance as yf
    HAS_YFINANCE = True
except ImportError:
    HAS_YFINANCE = False


class MetricsCalculator:
    """Calculate comprehensive trading metrics."""

    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate: Annual risk-free rate for Sharpe calculation (default 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = 252

    def calculate_all_metrics(
        self,
        equity_curve: pd.Series,
        trades: List[Dict],
        benchmark_returns: Optional[pd.Series] = None
    ) -> Dict:
        """
        Calculate all metrics for a backtest.

        Args:
            equity_curve: Series with datetime index and equity values
            trades: List of trade dictionaries with keys: entry_date, exit_date, pnl, etc.
            benchmark_returns: Optional benchmark returns series

        Returns:
            Dictionary with all calculated metrics
        """
        returns = equity_curve.pct_change().dropna()

        metrics = {
            # Return metrics
            **self.calculate_return_metrics(equity_curve, returns),

            # Risk-adjusted metrics
            **self.calculate_risk_adjusted_metrics(returns),

            # Drawdown metrics
            **self.calculate_drawdown_metrics(equity_curve),

            # Trade statistics
            **self.calculate_trade_statistics(trades),

            # Distribution metrics
            **self.calculate_distribution_metrics(returns),
        }

        # Add benchmark comparison if provided
        if benchmark_returns is not None:
            metrics.update(self.calculate_benchmark_metrics(returns, benchmark_returns))

        return metrics

    def calculate_return_metrics(
        self,
        equity_curve: pd.Series,
        returns: pd.Series
    ) -> Dict:
        """Calculate basic return metrics."""
        total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

        # Calculate trading days
        days = (equity_curve.index[-1] - equity_curve.index[0]).days
        years = days / 365.25

        # Annualized return (CAGR)
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'cagr': cagr,
            'cagr_pct': cagr * 100,
            'daily_return_mean': returns.mean(),
            'daily_return_std': returns.std(),
            'trading_days': len(equity_curve),
            'calendar_days': days,
            'years': years,
        }

    def calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk-adjusted performance metrics."""
        # Daily risk-free rate
        daily_rf = (1 + self.risk_free_rate) ** (1 / self.trading_days_per_year) - 1

        # Sharpe Ratio
        excess_returns = returns - daily_rf
        sharpe = np.sqrt(self.trading_days_per_year) * (
            excess_returns.mean() / excess_returns.std()
        ) if excess_returns.std() > 0 else 0

        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < daily_rf]
        downside_std = downside_returns.std()
        sortino = np.sqrt(self.trading_days_per_year) * (
            excess_returns.mean() / downside_std
        ) if downside_std > 0 else 0

        # Omega Ratio (probability-weighted ratio of gains vs losses)
        omega = self.calculate_omega_ratio(returns, daily_rf)

        # Calmar Ratio (return / max drawdown)
        annual_return = returns.mean() * self.trading_days_per_year
        max_dd = self.calculate_max_drawdown(
            (1 + returns).cumprod()
        )['max_drawdown']
        calmar = abs(annual_return / max_dd) if max_dd != 0 else 0

        return {
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'omega_ratio': omega,
            'calmar_ratio': calmar,
        }

    def calculate_omega_ratio(
        self,
        returns: pd.Series,
        threshold: float = 0.0
    ) -> float:
        """
        Calculate Omega ratio.

        Omega = (Probability-weighted gains) / (Probability-weighted losses)
        Values > 1 indicate more gains than losses relative to threshold.
        """
        excess = returns - threshold
        gains = excess[excess > 0].sum()
        losses = abs(excess[excess < 0].sum())

        return gains / losses if losses > 0 else np.inf

    def calculate_drawdown_metrics(self, equity_curve: pd.Series) -> Dict:
        """Calculate drawdown-related metrics."""
        dd_info = self.calculate_max_drawdown(equity_curve)

        # Calculate underwater periods (time spent in drawdown)
        running_max = equity_curve.expanding().max()
        underwater = equity_curve < running_max
        underwater_pct = underwater.sum() / len(equity_curve)

        return {
            'max_drawdown': dd_info['max_drawdown'],
            'max_drawdown_pct': dd_info['max_drawdown'] * 100,
            'max_drawdown_duration_days': dd_info['duration_days'],
            'max_drawdown_start': dd_info['start_date'],
            'max_drawdown_end': dd_info['end_date'],
            'underwater_pct': underwater_pct * 100,
        }

    def calculate_max_drawdown(self, equity_curve: pd.Series) -> Dict:
        """Calculate maximum drawdown and related info."""
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max

        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()

        # Find the peak before the max drawdown
        peak_idx = equity_curve[:max_dd_idx].idxmax()

        # Find recovery point (if any)
        recovery_idx = None
        if max_dd_idx < equity_curve.index[-1]:
            peak_value = equity_curve[peak_idx]
            future_equity = equity_curve[max_dd_idx:]
            recovery_points = future_equity[future_equity >= peak_value]
            if len(recovery_points) > 0:
                recovery_idx = recovery_points.index[0]

        # Calculate duration
        if recovery_idx:
            duration = (recovery_idx - peak_idx).days
        else:
            duration = (equity_curve.index[-1] - peak_idx).days

        return {
            'max_drawdown': max_dd,
            'start_date': peak_idx,
            'end_date': max_dd_idx,
            'recovery_date': recovery_idx,
            'duration_days': duration,
        }

    def calculate_trade_statistics(self, trades: List[Dict]) -> Dict:
        """Calculate detailed trade statistics."""
        if not trades:
            return self._empty_trade_stats()

        # Filter to closed trades only
        closed_trades = [t for t in trades if t.get('action') == 'EXIT']

        if not closed_trades:
            return self._empty_trade_stats()

        # Extract P&L values
        pnls = [t.get('pnl', 0) for t in closed_trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]

        # Win rate
        win_rate = len(winning_trades) / len(pnls) if pnls else 0

        # Profit factor
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else np.inf

        # Average wins/losses
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # Largest win/loss
        largest_win = max(pnls) if pnls else 0
        largest_loss = min(pnls) if pnls else 0

        # Average hold time
        hold_times = []
        for trade in closed_trades:
            if 'entry_date' in trade and 'exit_date' in trade:
                entry = pd.to_datetime(trade['entry_date'])
                exit_d = pd.to_datetime(trade['exit_date'])
                hold_times.append((exit_d - entry).days)

        avg_hold = np.mean(hold_times) if hold_times else 0

        # Consecutive wins/losses
        consecutive_wins = self._max_consecutive(pnls, lambda x: x > 0)
        consecutive_losses = self._max_consecutive(pnls, lambda x: x < 0)

        return {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'win_rate_pct': win_rate * 100,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_pnl': np.mean(pnls),
            'expectancy': expectancy,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_hold_days': avg_hold,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'total_pnl': sum(pnls),
        }

    def calculate_distribution_metrics(self, returns: pd.Series) -> Dict:
        """Calculate return distribution metrics."""
        return {
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'var_95': returns.quantile(0.05),  # 5% VaR
            'cvar_95': returns[returns <= returns.quantile(0.05)].mean(),  # CVaR
            'best_day': returns.max(),
            'worst_day': returns.min(),
        }

    def calculate_benchmark_metrics(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Dict:
        """Calculate metrics comparing strategy to benchmark."""
        # Align returns
        aligned = pd.DataFrame({
            'strategy': strategy_returns,
            'benchmark': benchmark_returns
        }).dropna()

        if len(aligned) == 0:
            return {}

        # Excess returns
        excess_returns = aligned['strategy'] - aligned['benchmark']

        # Information ratio
        info_ratio = np.sqrt(self.trading_days_per_year) * (
            excess_returns.mean() / excess_returns.std()
        ) if excess_returns.std() > 0 else 0

        # Beta
        covariance = np.cov(aligned['strategy'], aligned['benchmark'])[0, 1]
        benchmark_variance = aligned['benchmark'].var()
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0

        # Alpha (annualized)
        strategy_annual_return = aligned['strategy'].mean() * self.trading_days_per_year
        benchmark_annual_return = aligned['benchmark'].mean() * self.trading_days_per_year
        alpha = strategy_annual_return - (
            self.risk_free_rate + beta * (benchmark_annual_return - self.risk_free_rate)
        )

        # Correlation
        correlation = aligned.corr().iloc[0, 1]

        # Win rate vs benchmark
        outperformance = (aligned['strategy'] > aligned['benchmark']).sum()
        win_rate_vs_bench = outperformance / len(aligned)

        return {
            'alpha': alpha,
            'alpha_pct': alpha * 100,
            'beta': beta,
            'correlation': correlation,
            'information_ratio': info_ratio,
            'win_rate_vs_benchmark': win_rate_vs_bench,
            'win_rate_vs_benchmark_pct': win_rate_vs_bench * 100,
        }

    def get_benchmark_data(
        self,
        start_date: datetime,
        end_date: datetime,
        ticker: str = 'SPY'
    ) -> pd.Series:
        """
        Download benchmark data (default S&P 500 via SPY).

        Args:
            start_date: Start date
            end_date: End date
            ticker: Benchmark ticker (default SPY)

        Returns:
            Series of benchmark returns
        """
        if not HAS_YFINANCE:
            print("Warning: yfinance not installed. Benchmark data unavailable.")
            print("Install with: pip install yfinance")
            return pd.Series()

        try:
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False
            )

            if data.empty:
                return pd.Series()

            # Use adjusted close and calculate returns
            returns = data['Adj Close'].pct_change().dropna()
            return returns

        except Exception as e:
            print(f"Error downloading benchmark data: {e}")
            return pd.Series()

    def _empty_trade_stats(self) -> Dict:
        """Return empty trade statistics."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'win_rate_pct': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'avg_pnl': 0,
            'expectancy': 0,
            'largest_win': 0,
            'largest_loss': 0,
            'avg_hold_days': 0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'total_pnl': 0,
        }

    def _max_consecutive(self, values: List[float], condition) -> int:
        """Calculate maximum consecutive values meeting condition."""
        max_count = 0
        current_count = 0

        for val in values:
            if condition(val):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0

        return max_count
