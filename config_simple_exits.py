"""
Fixed Exit Configuration for X-Dividend Strategy

Key Changes:
1. NO stop losses (dividend drop is expected, not a "loss")
2. Exit based on TIME and TECHNICAL signals, not P&L
3. Simple, robust exit rules
"""

from dataclasses import dataclass


@dataclass
class SimpleDividendExitConfig:
    """
    Simple exit rules for dividend capture

    Philosophy:
    - Dividend drop is EXPECTED, not a loss
    - Hold through ex-div for mean reversion
    - Exit on time or technical signals
    - NO P&L-based stops
    """

    # Time-based exits (primary)
    min_holding_days: int = 1              # Minimum hold (after entry)
    max_holding_days: int = 15             # Maximum hold (safety)
    target_hold_after_exdiv: int = 5       # Target days after ex-div

    # Technical exits (secondary)
    use_rsi_exit: bool = True              # Exit if RSI > 70 (overbought)
    rsi_exit_threshold: float = 70.0       # RSI level to exit

    use_vwap_cross: bool = False           # Disabled (too unreliable)

    # Profit taking (optional, conservative)
    use_profit_target: bool = True
    profit_target_absolute: float = 0.05   # 5% (was 3%) - let winners run

    # Emergency exit only (very wide)
    use_emergency_stop: bool = True
    emergency_stop_pct: float = 0.10       # -10% (only for disasters)

    # Disabled features
    hard_stop_pct: float = None            # DISABLED - no hard stop
    use_dividend_stop: bool = False        # DISABLED - dividend drop expected
    trailing_stop_enabled: bool = False    # DISABLED - too complex


def apply_simple_exit_config():
    """Apply simple exit configuration to the strategy"""
    from config import exit_config

    # Override exit config with simple rules
    exit_config.min_holding_days = 1
    exit_config.max_holding_days = 15
    exit_config.hard_stop_pct = 1.0                    # Effectively disabled (100% loss)
    exit_config.use_dividend_stop = False              # CRITICAL: Don't exit on dividend drop
    exit_config.trailing_stop_enabled = False
    exit_config.profit_target_absolute = 0.05          # 5% take profit
    exit_config.use_vwap_cross = False
    exit_config.use_entry_plus_dividend = False        # Don't use this condition
    exit_config.dividend_adjustment_threshold_low = 0.0    # Disabled
    exit_config.dividend_adjustment_threshold_high = 2.0   # Disabled

    print("✅ Applied SIMPLE exit configuration")
    print("   Disabled: Hard stops, dividend stops, trailing stops")
    print("   Enabled: Time-based exits, 5% profit target, -10% emergency only")
    print("   Philosophy: Hold through dividend drop, exit on time/technical")
