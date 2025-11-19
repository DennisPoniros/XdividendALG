"""
Strategy Registry - Central repository for all available strategies

This module provides a registry system for managing multiple trading strategies,
making it easy to add new strategies without modifying dashboard code.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional
from pathlib import Path
import importlib
import inspect


@dataclass
class StrategyMetadata:
    """Metadata for a registered strategy."""

    name: str
    display_name: str
    description: str
    version: str
    author: str = "XDividend Team"

    # Strategy characteristics
    strategy_type: str = "dividend_capture"  # dividend_capture, momentum, mean_reversion, etc.
    requires_training: bool = False
    supports_live: bool = True

    # Parameters schema (for UI generation)
    parameters: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Execution details
    module_path: str = ""  # e.g., "strategy"
    class_name: str = ""   # e.g., "XDividendStrategy"
    backtester_path: str = ""  # e.g., "backtester"
    backtester_class: str = ""  # e.g., "Backtester"
    runner_path: str = ""  # e.g., "main"
    runner_function: str = "main"  # Function to call to run backtest

    # Config overrides
    config_overrides: Dict[str, Any] = field(default_factory=dict)

    # Tags for filtering/searching
    tags: List[str] = field(default_factory=list)


class StrategyRegistry:
    """
    Central registry for all trading strategies.

    Strategies can be registered statically (in code) or dynamically (plugins).
    """

    def __init__(self):
        """Initialize empty registry."""
        self.strategies: Dict[str, StrategyMetadata] = {}
        self._register_builtin_strategies()

    def _register_builtin_strategies(self):
        """Register all built-in strategies."""

        # Original X-Dividend Strategy
        self.register(StrategyMetadata(
            name="xdiv_original",
            display_name="X-Dividend (Original)",
            description="Original dividend capture strategy with fixed parameters (30% capture rate, weak mean reversion)",
            version="1.0.0",
            strategy_type="dividend_capture",
            requires_training=False,
            supports_live=True,
            module_path="strategy",
            class_name="XDividendStrategy",
            backtester_path="backtester",
            backtester_class="Backtester",
            runner_path="main",
            runner_function="main",
            parameters={
                "initial_capital": {
                    "type": "float",
                    "default": 100000,
                    "min": 10000,
                    "max": 10000000,
                    "description": "Starting capital for backtest"
                },
                "start_date": {
                    "type": "date",
                    "default": "2023-01-01",
                    "description": "Backtest start date"
                },
                "end_date": {
                    "type": "date",
                    "default": "2024-10-31",
                    "description": "Backtest end date"
                },
            },
            tags=["dividend", "capture", "baseline", "no-training"]
        ))

        # ML-Based X-Dividend Strategy (with problematic exits)
        self.register(StrategyMetadata(
            name="xdiv_ml",
            display_name="X-Dividend ML (Legacy)",
            description="ML-based strategy with training - WARNING: Has exit logic bug causing negative returns",
            version="2.0.0",
            strategy_type="dividend_capture",
            requires_training=True,
            supports_live=False,
            module_path="strategy_xdiv_ml",
            class_name="XDividendMLStrategy",
            backtester_path="backtester_xdiv_ml",
            backtester_class="XDividendMLBacktester",
            runner_path="run_xdiv_ml_backtest",
            runner_function="run_xdiv_ml_backtest",
            parameters={
                "train_start": {
                    "type": "date",
                    "default": "2018-01-01",
                    "description": "Training period start"
                },
                "train_end": {
                    "type": "date",
                    "default": "2022-12-31",
                    "description": "Training period end"
                },
                "test_start": {
                    "type": "date",
                    "default": "2023-01-01",
                    "description": "Test period start"
                },
                "test_end": {
                    "type": "date",
                    "default": "2024-10-31",
                    "description": "Test period end"
                },
                "initial_capital": {
                    "type": "float",
                    "default": 100000,
                    "min": 10000,
                    "max": 10000000,
                    "description": "Starting capital"
                },
            },
            config_overrides={},
            tags=["dividend", "machine-learning", "training", "deprecated"]
        ))

        # Fixed X-Dividend Strategy (RECOMMENDED)
        self.register(StrategyMetadata(
            name="xdiv_fixed",
            display_name="X-Dividend ML (Fixed) ⭐",
            description="ML-based strategy with FIXED exit logic - removes P&L-based stops, uses time-based exits. RECOMMENDED strategy.",
            version="3.0.0",
            strategy_type="dividend_capture",
            requires_training=True,
            supports_live=False,
            module_path="strategy_xdiv_ml_fixed",
            class_name="XDividendMLStrategyFixed",
            backtester_path="backtester_xdiv_ml",
            backtester_class="XDividendMLBacktester",
            runner_path="run_simple_backtest",
            runner_function="run_simple_backtest",
            parameters={
                "train_start": {
                    "type": "date",
                    "default": "2018-01-01",
                    "description": "Training period start"
                },
                "train_end": {
                    "type": "date",
                    "default": "2022-12-31",
                    "description": "Training period end"
                },
                "test_start": {
                    "type": "date",
                    "default": "2023-01-01",
                    "description": "Test period start"
                },
                "test_end": {
                    "type": "date",
                    "default": "2024-10-31",
                    "description": "Test period end"
                },
                "initial_capital": {
                    "type": "float",
                    "default": 100000,
                    "min": 10000,
                    "max": 10000000,
                    "description": "Starting capital"
                },
                "use_relaxed_screening": {
                    "type": "bool",
                    "default": True,
                    "description": "Use relaxed screening filters (recommended)"
                },
                "use_simple_exits": {
                    "type": "bool",
                    "default": True,
                    "description": "Use simple time-based exits (recommended)"
                },
            },
            config_overrides={
                "apply_relaxed_screening": True,
                "apply_simple_exits": True,
            },
            tags=["dividend", "machine-learning", "training", "fixed", "recommended"]
        ))

        # Post-Dividend Dip Strategy (ALTERNATIVE APPROACH)
        self.register(StrategyMetadata(
            name="post_div_dip",
            display_name="Post-Dividend Dip Buyer 🔄",
            description="INVERSE approach: Buy AFTER ex-div when price drops, sell at mean reversion. Avoids dividend taxation (15-37%), captures same mean reversion with simpler tax treatment.",
            version="1.0.0",
            strategy_type="mean_reversion",
            requires_training=False,
            supports_live=True,
            module_path="strategy_post_div_dip",
            class_name="PostDividendDipStrategy",
            backtester_path="backtester_post_div",
            backtester_class="PostDivDipBacktester",
            runner_path="run_post_div_backtest",
            runner_function="run_post_div_backtest",
            parameters={
                "start_date": {
                    "type": "date",
                    "default": "2023-01-01",
                    "description": "Backtest start date"
                },
                "end_date": {
                    "type": "date",
                    "default": "2024-10-31",
                    "description": "Backtest end date"
                },
                "initial_capital": {
                    "type": "float",
                    "default": 100000,
                    "min": 10000,
                    "max": 10000000,
                    "description": "Starting capital"
                },
            },
            config_overrides={},
            tags=["dividend", "mean-reversion", "tax-efficient", "alternative", "no-ml"]
        ))

    def register(self, strategy: StrategyMetadata):
        """Register a new strategy."""
        self.strategies[strategy.name] = strategy
        print(f"✅ Registered strategy: {strategy.display_name}")

    def get_strategy(self, name: str) -> Optional[StrategyMetadata]:
        """Get strategy by name."""
        return self.strategies.get(name)

    def list_strategies(
        self,
        strategy_type: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[StrategyMetadata]:
        """
        List all registered strategies with optional filtering.

        Args:
            strategy_type: Filter by strategy type
            tags: Filter by tags (any match)

        Returns:
            List of matching strategies
        """
        strategies = list(self.strategies.values())

        # Filter by type
        if strategy_type:
            strategies = [s for s in strategies if s.strategy_type == strategy_type]

        # Filter by tags
        if tags:
            strategies = [
                s for s in strategies
                if any(tag in s.tags for tag in tags)
            ]

        return strategies

    def get_strategy_parameters(self, name: str) -> Dict[str, Dict[str, Any]]:
        """Get parameter schema for a strategy."""
        strategy = self.get_strategy(name)
        if not strategy:
            return {}
        return strategy.parameters

    def get_default_parameters(self, name: str) -> Dict[str, Any]:
        """Get default parameter values for a strategy."""
        params_schema = self.get_strategy_parameters(name)
        return {
            param_name: param_def.get("default")
            for param_name, param_def in params_schema.items()
        }


# Global registry instance
_global_registry = None


def get_registry() -> StrategyRegistry:
    """Get the global strategy registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = StrategyRegistry()
    return _global_registry


def register_strategy(strategy: StrategyMetadata):
    """Convenience function to register a strategy."""
    registry = get_registry()
    registry.register(strategy)


def list_strategies(**kwargs) -> List[StrategyMetadata]:
    """Convenience function to list strategies."""
    registry = get_registry()
    return registry.list_strategies(**kwargs)
