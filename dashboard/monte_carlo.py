"""
Monte Carlo simulation for backtesting robustness analysis.

Resamples trades to test if results are due to skill or luck.
Calculates confidence intervals and probability of outcomes.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class MonteCarloSimulator:
    """Monte Carlo simulation for trade resampling."""

    def __init__(self, n_simulations: int = 1000, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo simulator.

        Args:
            n_simulations: Number of simulation runs
            random_seed: Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        if random_seed is not None:
            np.random.seed(random_seed)

    def run_simulation(
        self,
        trades: List[Dict],
        initial_capital: float = 100000
    ) -> Dict:
        """
        Run Monte Carlo simulation by resampling trades.

        Args:
            trades: List of closed trades with P&L
            initial_capital: Starting capital

        Returns:
            Dictionary with simulation results
        """
        # Extract P&Ls from closed trades
        closed_trades = [t for t in trades if t.get('action') == 'EXIT']

        if len(closed_trades) < 10:
            return self._empty_results("Need at least 10 trades for meaningful simulation")

        pnls = np.array([t.get('pnl', 0) for t in closed_trades])
        n_trades = len(pnls)

        # Run simulations
        final_equities = []
        max_drawdowns = []
        sharpe_ratios = []
        win_rates = []
        profit_factors = []

        for _ in range(self.n_simulations):
            # Resample trades with replacement
            resampled_pnls = np.random.choice(pnls, size=n_trades, replace=True)

            # Calculate equity curve
            equity_curve = initial_capital + np.cumsum(resampled_pnls)
            equity_curve = np.insert(equity_curve, 0, initial_capital)

            # Final equity
            final_equity = equity_curve[-1]
            final_equities.append(final_equity)

            # Max drawdown
            running_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - running_max) / running_max
            max_dd = drawdown.min()
            max_drawdowns.append(max_dd)

            # Win rate
            wins = (resampled_pnls > 0).sum()
            win_rate = wins / n_trades
            win_rates.append(win_rate)

            # Sharpe ratio (simplified)
            if len(resampled_pnls) > 1:
                returns = resampled_pnls / initial_capital
                sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() > 0 else 0
                sharpe_ratios.append(sharpe)

            # Profit factor
            total_wins = resampled_pnls[resampled_pnls > 0].sum()
            total_losses = abs(resampled_pnls[resampled_pnls < 0].sum())
            pf = total_wins / total_losses if total_losses > 0 else np.inf
            profit_factors.append(pf if pf != np.inf else 10)  # Cap at 10 for stats

        # Calculate statistics
        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)
        sharpe_ratios = np.array(sharpe_ratios)
        win_rates = np.array(win_rates)
        profit_factors = np.array(profit_factors)

        # Calculate actual values from original trades
        actual_equity = initial_capital + pnls.sum()
        actual_running_max = np.maximum.accumulate(
            np.insert(initial_capital + np.cumsum(pnls), 0, initial_capital)
        )
        actual_equity_curve = np.insert(initial_capital + np.cumsum(pnls), 0, initial_capital)
        actual_drawdown = (actual_equity_curve - actual_running_max) / actual_running_max
        actual_max_dd = actual_drawdown.min()
        actual_win_rate = (pnls > 0).sum() / len(pnls)
        actual_returns = pnls / initial_capital
        actual_sharpe = np.sqrt(252) * (actual_returns.mean() / actual_returns.std()) if actual_returns.std() > 0 else 0

        return {
            # Simulated distributions
            'final_equities': final_equities,
            'max_drawdowns': max_drawdowns,
            'sharpe_ratios': sharpe_ratios,
            'win_rates': win_rates,
            'profit_factors': profit_factors,

            # Statistics
            'equity_percentiles': {
                '5th': np.percentile(final_equities, 5),
                '25th': np.percentile(final_equities, 25),
                '50th': np.percentile(final_equities, 50),
                '75th': np.percentile(final_equities, 75),
                '95th': np.percentile(final_equities, 95),
                'mean': final_equities.mean(),
                'std': final_equities.std(),
            },
            'drawdown_percentiles': {
                '5th': np.percentile(max_drawdowns, 5),
                '25th': np.percentile(max_drawdowns, 25),
                '50th': np.percentile(max_drawdowns, 50),
                '75th': np.percentile(max_drawdowns, 75),
                '95th': np.percentile(max_drawdowns, 95),
                'mean': max_drawdowns.mean(),
                'std': max_drawdowns.std(),
            },
            'sharpe_percentiles': {
                '5th': np.percentile(sharpe_ratios, 5),
                '25th': np.percentile(sharpe_ratios, 25),
                '50th': np.percentile(sharpe_ratios, 50),
                '75th': np.percentile(sharpe_ratios, 75),
                '95th': np.percentile(sharpe_ratios, 95),
                'mean': sharpe_ratios.mean(),
                'std': sharpe_ratios.std(),
            },

            # Actual values
            'actual_equity': actual_equity,
            'actual_max_dd': actual_max_dd,
            'actual_sharpe': actual_sharpe,
            'actual_win_rate': actual_win_rate,

            # Probabilities
            'prob_profit': (final_equities > initial_capital).sum() / self.n_simulations,
            'prob_10pct_return': (final_equities > initial_capital * 1.1).sum() / self.n_simulations,
            'prob_20pct_return': (final_equities > initial_capital * 1.2).sum() / self.n_simulations,
            'prob_drawdown_gt_20pct': (max_drawdowns < -0.20).sum() / self.n_simulations,
            'prob_sharpe_gt_1': (sharpe_ratios > 1.0).sum() / self.n_simulations,

            # Percentile ranks of actual results
            'actual_equity_percentile': (final_equities < actual_equity).sum() / self.n_simulations * 100,
            'actual_sharpe_percentile': (sharpe_ratios < actual_sharpe).sum() / self.n_simulations * 100,

            # Metadata
            'n_simulations': self.n_simulations,
            'n_trades': n_trades,
            'initial_capital': initial_capital,
        }

    def calculate_confidence_intervals(
        self,
        simulation_results: Dict,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Calculate confidence intervals from simulation results.

        Args:
            simulation_results: Results from run_simulation
            confidence_level: Confidence level (default 95%)

        Returns:
            Dictionary with confidence intervals
        """
        alpha = 1 - confidence_level
        lower_percentile = alpha / 2 * 100
        upper_percentile = (1 - alpha / 2) * 100

        final_equities = simulation_results['final_equities']
        max_drawdowns = simulation_results['max_drawdowns']
        sharpe_ratios = simulation_results['sharpe_ratios']

        return {
            'confidence_level': confidence_level,
            'equity_ci': (
                np.percentile(final_equities, lower_percentile),
                np.percentile(final_equities, upper_percentile)
            ),
            'drawdown_ci': (
                np.percentile(max_drawdowns, lower_percentile),
                np.percentile(max_drawdowns, upper_percentile)
            ),
            'sharpe_ci': (
                np.percentile(sharpe_ratios, lower_percentile),
                np.percentile(sharpe_ratios, upper_percentile)
            ),
        }

    def assess_robustness(self, simulation_results: Dict) -> Dict:
        """
        Assess strategy robustness based on Monte Carlo results.

        Args:
            simulation_results: Results from run_simulation

        Returns:
            Dictionary with robustness assessment
        """
        actual_equity_pct = simulation_results['actual_equity_percentile']
        actual_sharpe_pct = simulation_results['actual_sharpe_percentile']
        prob_profit = simulation_results['prob_profit']
        prob_sharpe_gt_1 = simulation_results['prob_sharpe_gt_1']

        # Assess robustness
        robustness_score = 0
        robustness_factors = []

        # Factor 1: Actual results vs simulation distribution
        if actual_equity_pct > 75:
            robustness_score += 25
            robustness_factors.append("Actual equity in top quartile")
        elif actual_equity_pct > 50:
            robustness_score += 15
            robustness_factors.append("Actual equity above median")
        elif actual_equity_pct > 25:
            robustness_score += 5
            robustness_factors.append("Actual equity below median")
        else:
            robustness_factors.append("WARNING: Actual equity in bottom quartile")

        # Factor 2: Probability of profit
        if prob_profit > 0.90:
            robustness_score += 25
            robustness_factors.append(f"High profit probability ({prob_profit:.1%})")
        elif prob_profit > 0.75:
            robustness_score += 15
            robustness_factors.append(f"Good profit probability ({prob_profit:.1%})")
        elif prob_profit > 0.60:
            robustness_score += 5
            robustness_factors.append(f"Moderate profit probability ({prob_profit:.1%})")
        else:
            robustness_factors.append(f"WARNING: Low profit probability ({prob_profit:.1%})")

        # Factor 3: Sharpe consistency
        if prob_sharpe_gt_1 > 0.75:
            robustness_score += 25
            robustness_factors.append(f"Consistent Sharpe > 1 ({prob_sharpe_gt_1:.1%})")
        elif prob_sharpe_gt_1 > 0.50:
            robustness_score += 15
            robustness_factors.append(f"Good Sharpe consistency ({prob_sharpe_gt_1:.1%})")
        elif prob_sharpe_gt_1 > 0.30:
            robustness_score += 5
            robustness_factors.append(f"Moderate Sharpe consistency ({prob_sharpe_gt_1:.1%})")
        else:
            robustness_factors.append(f"WARNING: Low Sharpe consistency ({prob_sharpe_gt_1:.1%})")

        # Factor 4: Result stability (low variance)
        equity_cv = simulation_results['equity_percentiles']['std'] / simulation_results['equity_percentiles']['mean']
        if equity_cv < 0.15:
            robustness_score += 25
            robustness_factors.append("Very stable results")
        elif equity_cv < 0.30:
            robustness_score += 15
            robustness_factors.append("Stable results")
        elif equity_cv < 0.50:
            robustness_score += 5
            robustness_factors.append("Moderately stable results")
        else:
            robustness_factors.append("WARNING: Highly variable results")

        # Overall assessment
        if robustness_score >= 80:
            assessment = "EXCELLENT - Strategy is highly robust"
        elif robustness_score >= 60:
            assessment = "GOOD - Strategy shows good robustness"
        elif robustness_score >= 40:
            assessment = "MODERATE - Strategy has some robustness concerns"
        else:
            assessment = "POOR - Strategy may be overfitted or luck-driven"

        return {
            'robustness_score': robustness_score,
            'assessment': assessment,
            'factors': robustness_factors,
            'equity_cv': equity_cv,
        }

    def _empty_results(self, message: str) -> Dict:
        """Return empty results with message."""
        return {
            'error': message,
            'final_equities': np.array([]),
            'max_drawdowns': np.array([]),
            'sharpe_ratios': np.array([]),
            'win_rates': np.array([]),
            'profit_factors': np.array([]),
        }
