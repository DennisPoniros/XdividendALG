"""
Backtest Executor - Run backtests from dashboard

Handles execution of backtests in background threads with progress tracking.
"""

import threading
import queue
import traceback
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from enum import Enum
import sys
from pathlib import Path
import importlib


class BacktestStatus(Enum):
    """Status of a backtest run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BacktestRun:
    """Represents a single backtest run."""

    run_id: str
    strategy_name: str
    parameters: Dict[str, Any]
    status: BacktestStatus = BacktestStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress: float = 0.0  # 0.0 to 1.0
    progress_message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    logs: list = field(default_factory=list)


class BacktestExecutor:
    """
    Execute backtests in background threads.

    Features:
    - Non-blocking execution
    - Progress tracking
    - Log capture
    - Result storage
    - Multiple concurrent runs
    """

    def __init__(self):
        """Initialize executor."""
        self.runs: Dict[str, BacktestRun] = {}
        self.run_counter = 0
        self.max_concurrent_runs = 1  # Only one backtest at a time (resource-intensive)

    def create_run(
        self,
        strategy_name: str,
        parameters: Dict[str, Any]
    ) -> str:
        """
        Create a new backtest run.

        Args:
            strategy_name: Name of strategy to run
            parameters: Strategy parameters

        Returns:
            Run ID
        """
        self.run_counter += 1
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self.run_counter}"

        run = BacktestRun(
            run_id=run_id,
            strategy_name=strategy_name,
            parameters=parameters
        )

        self.runs[run_id] = run
        return run_id

    def execute_run(
        self,
        run_id: str,
        strategy_metadata: Any,  # StrategyMetadata
        config_overrides: Optional[Dict[str, Any]] = None
    ):
        """
        Execute a backtest run in background thread.

        Args:
            run_id: Run ID
            strategy_metadata: Strategy metadata from registry
            config_overrides: Additional config overrides
        """
        if run_id not in self.runs:
            raise ValueError(f"Run {run_id} not found")

        # Check if we're at concurrent limit
        running_count = len([r for r in self.runs.values() if r.status == BacktestStatus.RUNNING])
        if running_count >= self.max_concurrent_runs:
            raise RuntimeError("Maximum concurrent backtests reached. Please wait for current run to finish.")

        # Start execution thread
        thread = threading.Thread(
            target=self._run_backtest_thread,
            args=(run_id, strategy_metadata, config_overrides or {}),
            daemon=True
        )
        thread.start()

    def _run_backtest_thread(
        self,
        run_id: str,
        strategy_metadata: Any,
        config_overrides: Dict[str, Any]
    ):
        """Run backtest in background thread."""
        run = self.runs[run_id]

        try:
            # Update status
            run.status = BacktestStatus.RUNNING
            run.start_time = datetime.now()
            run.progress = 0.0
            run.progress_message = "Initializing..."
            run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Starting backtest: {strategy_metadata.display_name}")

            # Add parent directory to path if needed
            parent_dir = str(Path(__file__).parent.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            # Import strategy runner module
            run.progress = 0.1
            run.progress_message = "Loading strategy module..."
            run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Importing {strategy_metadata.runner_path}")

            try:
                runner_module = importlib.import_module(strategy_metadata.runner_path)
                runner_function = getattr(runner_module, strategy_metadata.runner_function)
            except (ImportError, AttributeError) as e:
                raise ImportError(f"Could not import {strategy_metadata.runner_path}.{strategy_metadata.runner_function}: {e}")

            # Apply configuration overrides
            run.progress = 0.2
            run.progress_message = "Applying configuration..."
            run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Applying configuration overrides")

            # Apply relaxed screening if needed
            if config_overrides.get('apply_relaxed_screening'):
                try:
                    from config_relaxed import use_relaxed_screening
                    use_relaxed_screening()
                    run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Applied relaxed screening")
                except ImportError:
                    run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Could not import relaxed screening")

            # Apply simple exits if needed
            if config_overrides.get('apply_simple_exits'):
                try:
                    from config_simple_exits import apply_simple_exit_config
                    apply_simple_exit_config()
                    run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Applied simple exit configuration")
                except ImportError:
                    run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Could not import simple exits")

            # Update config with user parameters (if strategy supports it)
            if run.parameters:
                run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Applying user parameters:")
                for key, value in run.parameters.items():
                    run.logs.append(f"    {key}: {value}")

            # Run the backtest
            run.progress = 0.3
            run.progress_message = "Running backtest..."
            run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Executing backtest...")

            # Call the runner function
            # Note: Most runner functions don't return anything, they save to files
            result = runner_function()

            # Mark as complete
            run.progress = 1.0
            run.progress_message = "Completed"
            run.status = BacktestStatus.COMPLETED
            run.end_time = datetime.now()
            run.result = result if result else {"status": "completed"}
            run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Backtest completed successfully")

            # Calculate runtime
            runtime = (run.end_time - run.start_time).total_seconds()
            run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Runtime: {runtime:.1f} seconds")

        except Exception as e:
            # Handle errors
            run.status = BacktestStatus.FAILED
            run.end_time = datetime.now()
            run.error = str(e)
            run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ ERROR: {str(e)}")
            run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] Traceback:")
            for line in traceback.format_exc().split('\n'):
                if line.strip():
                    run.logs.append(f"    {line}")

    def get_run(self, run_id: str) -> Optional[BacktestRun]:
        """Get run by ID."""
        return self.runs.get(run_id)

    def get_recent_runs(self, limit: int = 10) -> list:
        """Get recent runs."""
        runs = sorted(
            self.runs.values(),
            key=lambda r: r.start_time or datetime.min,
            reverse=True
        )
        return runs[:limit]

    def cancel_run(self, run_id: str):
        """
        Cancel a running backtest.

        Note: Cannot actually stop thread, but marks as cancelled.
        """
        if run_id in self.runs:
            run = self.runs[run_id]
            if run.status == BacktestStatus.RUNNING:
                run.status = BacktestStatus.CANCELLED
                run.end_time = datetime.now()
                run.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠️  Run cancelled by user")

    def clear_old_runs(self, keep_recent: int = 20):
        """Clear old runs to save memory."""
        runs_sorted = sorted(
            self.runs.items(),
            key=lambda x: x[1].start_time or datetime.min,
            reverse=True
        )

        # Keep only recent runs
        if len(runs_sorted) > keep_recent:
            to_remove = runs_sorted[keep_recent:]
            for run_id, _ in to_remove:
                del self.runs[run_id]


# Global executor instance
_global_executor = None


def get_executor() -> BacktestExecutor:
    """Get the global backtest executor instance."""
    global _global_executor
    if _global_executor is None:
        _global_executor = BacktestExecutor()
    return _global_executor
