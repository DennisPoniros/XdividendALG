"""
Relaxed Screening Configuration for X-Dividend ML Strategy

Philosophy:
- Filter for BASIC quality (liquidity, dividend history)
- Let STRATEGY filters do the work (RSI, Z-score, momentum)
- Focus on dividend capture + mean reversion, not fundamental perfection
"""

from dataclasses import dataclass
from typing import List


@dataclass
class RelaxedScreeningConfig:
    """
    Minimal, logical filters - let the strategy handle selection
    """

    # BASIC LIQUIDITY (essential)
    min_market_cap: float = 500e6          # $500M (was $1B) - mid-cap+
    min_avg_volume: int = 100_000          # 100k shares/day (was 500k)

    # DIVIDEND REQUIREMENTS (core to strategy)
    min_dividend_yield: float = 0.01       # 1% (was 2%) - just has dividend
    max_dividend_yield: float = 0.20       # 20% (was 8%) - allow high-yield
    min_dividend_history_years: int = 2    # 2 years (was 5) - some history

    # EX-DIVIDEND WINDOW (timing)
    min_days_to_ex_div: int = 2            # 2 days (was 3)
    max_days_to_ex_div: int = 25           # 25 days (was 20) - wider window

    # QUALITY SCORE (relaxed)
    min_quality_score: float = 40.0        # 40/100 (was 70) - basic quality

    # PAYOUT RATIO (logical bounds)
    optimal_payout_min: float = 0.20       # 20% (was 40%)
    optimal_payout_max: float = 0.80       # 80% (was 60%)
    max_acceptable_payout: float = 1.5     # 150% (was 80%) - allow high for REITs

    # FINANCIAL HEALTH (minimal requirements)
    max_debt_to_equity: float = 2.0        # 2.0 (was 0.5) - very relaxed
    min_roe: float = -0.50                 # -50% (was 12%) - just not bankrupt
    max_pe_ratio: float = 100.0            # 100 (was 25) - allow growth stocks

    # TECHNICAL FILTERS (wide bounds - let strategy decide)
    min_rsi: float = 20                    # 20 (was 30) - more oversold
    max_rsi: float = 80                    # 80 (was 70) - less restrictive
    require_positive_momentum: bool = False # Don't require (was True)
    max_beta: float = 3.0                  # 3.0 (was 1.2) - allow volatile stocks
    max_short_interest: float = 0.30       # 30% (was 10%) - allow heavily shorted


@dataclass
class MinimalScreeningConfig:
    """
    ABSOLUTE MINIMUM filters - maximum opportunities
    Use this if relaxed is still too strict
    """

    # Only essential filters
    min_market_cap: float = 250e6          # $250M - small-cap OK
    min_avg_volume: int = 50_000           # 50k shares/day

    # Dividend exists
    min_dividend_yield: float = 0.005      # 0.5% - any dividend
    max_dividend_yield: float = 0.50       # 50% - even extreme yields
    min_dividend_history_years: int = 1    # 1 year only

    # Wide timing window
    min_days_to_ex_div: int = 1
    max_days_to_ex_div: int = 30

    # No quality filter
    min_quality_score: float = 0.0         # Accept all

    # No fundamental filters
    optimal_payout_min: float = 0.0
    optimal_payout_max: float = 2.0
    max_acceptable_payout: float = 5.0
    max_debt_to_equity: float = 10.0
    min_roe: float = -1.0                  # Allow losses
    max_pe_ratio: float = 1000.0

    # No technical screening filters
    min_rsi: float = 0
    max_rsi: float = 100
    require_positive_momentum: bool = False
    max_beta: float = 10.0
    max_short_interest: float = 1.0


# Helper functions to switch configs
def use_relaxed_screening():
    """Switch to relaxed screening (recommended)"""
    from config import screening_config

    # Override with relaxed values
    screening_config.min_market_cap = 500e6
    screening_config.min_avg_volume = 100_000
    screening_config.min_dividend_yield = 0.01
    screening_config.max_dividend_yield = 0.20
    screening_config.min_dividend_history_years = 2
    screening_config.min_days_to_ex_div = 2
    screening_config.max_days_to_ex_div = 25
    screening_config.min_quality_score = 40.0
    screening_config.optimal_payout_min = 0.20
    screening_config.optimal_payout_max = 0.80
    screening_config.max_acceptable_payout = 1.5
    screening_config.max_debt_to_equity = 2.0
    screening_config.min_roe = -0.50
    screening_config.max_pe_ratio = 100.0
    screening_config.min_rsi = 20
    screening_config.max_rsi = 80
    screening_config.require_positive_momentum = False
    screening_config.max_beta = 3.0
    screening_config.max_short_interest = 0.30

    print("✅ Switched to RELAXED screening")
    print(f"   Min ROE: {screening_config.min_roe*100:.0f}% (was 12%)")
    print(f"   Max P/E: {screening_config.max_pe_ratio:.0f} (was 25)")
    print(f"   Min Quality: {screening_config.min_quality_score:.0f}/100 (was 70)")
    print(f"   Min Div Yield: {screening_config.min_dividend_yield*100:.1f}% (was 2%)")


def use_minimal_screening():
    """Switch to minimal screening (most opportunities)"""
    from config import screening_config

    # Override with minimal values
    screening_config.min_market_cap = 250e6
    screening_config.min_avg_volume = 50_000
    screening_config.min_dividend_yield = 0.005
    screening_config.max_dividend_yield = 0.50
    screening_config.min_dividend_history_years = 1
    screening_config.min_days_to_ex_div = 1
    screening_config.max_days_to_ex_div = 30
    screening_config.min_quality_score = 0.0
    screening_config.optimal_payout_min = 0.0
    screening_config.optimal_payout_max = 2.0
    screening_config.max_acceptable_payout = 5.0
    screening_config.max_debt_to_equity = 10.0
    screening_config.min_roe = -1.0
    screening_config.max_pe_ratio = 1000.0
    screening_config.min_rsi = 0
    screening_config.max_rsi = 100
    screening_config.require_positive_momentum = False
    screening_config.max_beta = 10.0
    screening_config.max_short_interest = 1.0

    print("✅ Switched to MINIMAL screening")
    print(f"   Filtering only for: liquidity + dividend existence")
    print(f"   All quality filtering done by strategy signals")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("SCREENING CONFIGURATION OPTIONS")
    print("="*80)

    print("\nCurrent (STRICT):")
    print("  - Only 2 trades in 11 months")
    print("  - Filters out 99.9% of candidates")
    print("  - Too many arbitrary thresholds")

    print("\nRelaxed (RECOMMENDED):")
    print("  - Basic quality filters")
    print("  - Focuses on liquidity + dividend")
    print("  - Lets strategy signals do filtering")
    print("  - Expected: 50-150 trades/year")

    print("\nMinimal (MAXIMUM OPPORTUNITIES):")
    print("  - Only essential filters")
    print("  - Strategy does all selection")
    print("  - Expected: 100-300 trades/year")

    print("\nTo use in your code:")
    print("  from config_relaxed import use_relaxed_screening")
    print("  use_relaxed_screening()  # Before running backtest")
    print("="*80)
