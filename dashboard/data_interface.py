"""
Data interface for connecting dashboard to backtester.

Supports:
- Live mode: Real-time data streaming during backtest
- Replay mode: Load completed backtest results
- Data caching and efficient updates
"""

import pandas as pd
import numpy as np
import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import threading
import time


class BacktestDataInterface:
    """Interface to access backtest data for dashboard."""

    def __init__(
        self,
        output_dir: str = "/mnt/user-data/outputs",
        mode: str = "replay"
    ):
        """
        Initialize data interface.

        Args:
            output_dir: Directory containing backtest outputs
            mode: 'live' or 'replay'
        """
        self.output_dir = Path(output_dir)
        self.mode = mode
        self.cache = {}
        self.live_data_lock = threading.Lock()

        # Live mode data structures
        self.live_equity_curve = pd.Series(dtype=float)
        self.live_trades = []
        self.live_positions = {}
        self.live_position_values = pd.Series(dtype=float)

    def load_backtest_results(
        self,
        backtest_name: Optional[str] = None
    ) -> Dict:
        """
        Load completed backtest results.

        Args:
            backtest_name: Name of backtest (defaults to latest)

        Returns:
            Dictionary with equity_curve, trades, metrics, etc.
        """
        if backtest_name:
            # Load specific backtest
            results_file = self.output_dir / f"{backtest_name}_results.pkl"
        else:
            # Find latest results file
            results_files = list(self.output_dir.glob("*_results.pkl"))
            if not results_files:
                # Try loading from CSV if pickle doesn't exist
                return self._load_from_csv()

            results_file = max(results_files, key=lambda p: p.stat().st_mtime)

        if results_file.exists():
            with open(results_file, 'rb') as f:
                return pickle.load(f)
        else:
            return self._load_from_csv()

    def _load_from_csv(self) -> Dict:
        """Load backtest data from CSV files (fallback method)."""
        # Load trade log
        trade_files = list(self.output_dir.glob("*trade_log.csv"))
        if not trade_files:
            return self._empty_results()

        trade_file = max(trade_files, key=lambda p: p.stat().st_mtime)
        trades_df = pd.read_csv(trade_file)

        # Convert to list of dicts
        trades = trades_df.to_dict('records')

        # Reconstruct equity curve from trades
        equity_curve = self._reconstruct_equity_curve(trades)

        return {
            'equity_curve': equity_curve,
            'trades': trades,
            'positions': {},
            'position_values': pd.Series(dtype=float),
            'config': {},
        }

    def _reconstruct_equity_curve(
        self,
        trades: List[Dict],
        initial_capital: float = 100000
    ) -> pd.Series:
        """Reconstruct equity curve from trade log."""
        if not trades:
            return pd.Series(dtype=float)

        # Create timeline
        dates = []
        for trade in trades:
            if 'date' in trade:
                dates.append(pd.to_datetime(trade['date']))

        if not dates:
            return pd.Series(dtype=float)

        # Create date range
        date_range = pd.date_range(min(dates), max(dates), freq='D')

        # Calculate equity at each trade
        equity = initial_capital
        equity_by_date = {}

        for trade in trades:
            if trade.get('action') == 'EXIT' and 'pnl' in trade:
                date = pd.to_datetime(trade['date'])
                equity += trade['pnl']
                equity_by_date[date] = equity

        # Fill in missing dates
        equity_curve = pd.Series(index=date_range, dtype=float)
        equity_curve.iloc[0] = initial_capital

        for date in date_range:
            if date in equity_by_date:
                equity_curve[date] = equity_by_date[date]
            else:
                # Forward fill
                prev_date = date - pd.Timedelta(days=1)
                if prev_date in equity_curve.index:
                    equity_curve[date] = equity_curve[prev_date]

        return equity_curve.ffill()

    def get_equity_curve(self) -> pd.Series:
        """Get current equity curve."""
        if self.mode == "live":
            with self.live_data_lock:
                return self.live_equity_curve.copy()
        else:
            if 'equity_curve' not in self.cache:
                results = self.load_backtest_results()
                self.cache['equity_curve'] = results.get('equity_curve', pd.Series())

            return self.cache['equity_curve']

    def get_trades(self) -> List[Dict]:
        """Get list of all trades."""
        if self.mode == "live":
            with self.live_data_lock:
                return self.live_trades.copy()
        else:
            if 'trades' not in self.cache:
                results = self.load_backtest_results()
                self.cache['trades'] = results.get('trades', [])

            return self.cache['trades']

    def get_position_values(self) -> pd.Series:
        """Get time series of position values."""
        if self.mode == "live":
            with self.live_data_lock:
                return self.live_position_values.copy()
        else:
            if 'position_values' not in self.cache:
                results = self.load_backtest_results()
                self.cache['position_values'] = results.get('position_values', pd.Series())

            return self.cache['position_values']

    def get_current_positions(self) -> Dict:
        """Get current open positions."""
        if self.mode == "live":
            with self.live_data_lock:
                return self.live_positions.copy()
        else:
            results = self.load_backtest_results()
            return results.get('positions', {})

    def update_live_data(
        self,
        date: datetime,
        equity: float,
        trades: Optional[List[Dict]] = None,
        positions: Optional[Dict] = None,
        position_value: Optional[float] = None
    ):
        """
        Update live data during backtest.

        Args:
            date: Current date
            equity: Current equity value
            trades: New trades (if any)
            positions: Current positions
            position_value: Total position value
        """
        with self.live_data_lock:
            # Update equity curve
            self.live_equity_curve[pd.to_datetime(date)] = equity

            # Add new trades
            if trades:
                self.live_trades.extend(trades)

            # Update positions
            if positions is not None:
                self.live_positions = positions.copy()

            # Update position values
            if position_value is not None:
                self.live_position_values[pd.to_datetime(date)] = position_value

    def clear_live_data(self):
        """Clear all live data (start fresh backtest)."""
        with self.live_data_lock:
            self.live_equity_curve = pd.Series(dtype=float)
            self.live_trades = []
            self.live_positions = {}
            self.live_position_values = pd.Series(dtype=float)

    def save_results(
        self,
        name: str,
        equity_curve: pd.Series,
        trades: List[Dict],
        positions: Dict,
        position_values: pd.Series,
        config: Dict
    ):
        """
        Save backtest results to file.

        Args:
            name: Backtest name
            equity_curve: Equity curve series
            trades: List of trades
            positions: Final positions
            position_values: Position values series
            config: Configuration used
        """
        results = {
            'equity_curve': equity_curve,
            'trades': trades,
            'positions': positions,
            'position_values': position_values,
            'config': config,
            'timestamp': datetime.now(),
        }

        # Save as pickle
        results_file = self.output_dir / f"{name}_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)

        print(f"Results saved to {results_file}")

    def list_available_backtests(self) -> List[Dict]:
        """
        List all available backtest results.

        Returns:
            List of dicts with name, date, file path
        """
        results_files = list(self.output_dir.glob("*_results.pkl"))

        backtests = []
        for file in results_files:
            backtests.append({
                'name': file.stem.replace('_results', ''),
                'path': str(file),
                'modified': datetime.fromtimestamp(file.stat().st_mtime),
                'size': file.stat().st_size,
            })

        # Sort by modified date (newest first)
        backtests.sort(key=lambda x: x['modified'], reverse=True)

        return backtests

    def get_trade_dataframe(self) -> pd.DataFrame:
        """Get trades as a pandas DataFrame."""
        trades = self.get_trades()

        if not trades:
            return pd.DataFrame()

        df = pd.DataFrame(trades)

        # Convert date columns
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])

        if 'entry_date' in df.columns:
            df['entry_date'] = pd.to_datetime(df['entry_date'])

        if 'exit_date' in df.columns:
            df['exit_date'] = pd.to_datetime(df['exit_date'])

        return df

    def get_summary_stats(self) -> Dict:
        """Get quick summary statistics."""
        equity_curve = self.get_equity_curve()
        trades = self.get_trades()

        if len(equity_curve) == 0:
            return self._empty_stats()

        closed_trades = [t for t in trades if t.get('action') == 'EXIT']
        pnls = [t.get('pnl', 0) for t in closed_trades]

        return {
            'current_equity': equity_curve.iloc[-1] if len(equity_curve) > 0 else 0,
            'initial_equity': equity_curve.iloc[0] if len(equity_curve) > 0 else 0,
            'total_trades': len(closed_trades),
            'winning_trades': len([p for p in pnls if p > 0]),
            'losing_trades': len([p for p in pnls if p < 0]),
            'total_pnl': sum(pnls),
            'current_positions': len(self.get_current_positions()),
        }

    def _empty_results(self) -> Dict:
        """Return empty results structure."""
        return {
            'equity_curve': pd.Series(dtype=float),
            'trades': [],
            'positions': {},
            'position_values': pd.Series(dtype=float),
            'config': {},
        }

    def _empty_stats(self) -> Dict:
        """Return empty statistics."""
        return {
            'current_equity': 0,
            'initial_equity': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'current_positions': 0,
        }


class LiveBacktestStreamer:
    """Stream data from backtester to dashboard in real-time."""

    def __init__(self, data_interface: BacktestDataInterface):
        """
        Initialize live streamer.

        Args:
            data_interface: Data interface instance
        """
        self.data_interface = data_interface
        self.is_running = False
        self.update_interval = 1.0  # seconds

    def start(self):
        """Start streaming data."""
        self.is_running = True
        self.data_interface.clear_live_data()

    def stop(self):
        """Stop streaming data."""
        self.is_running = False

    def update(
        self,
        date: datetime,
        equity: float,
        trades: Optional[List[Dict]] = None,
        positions: Optional[Dict] = None,
        position_value: Optional[float] = None
    ):
        """
        Update data (called from backtester).

        Args:
            date: Current simulation date
            equity: Current equity
            trades: New trades
            positions: Current positions
            position_value: Total position value
        """
        if self.is_running:
            self.data_interface.update_live_data(
                date=date,
                equity=equity,
                trades=trades,
                positions=positions,
                position_value=position_value
            )
