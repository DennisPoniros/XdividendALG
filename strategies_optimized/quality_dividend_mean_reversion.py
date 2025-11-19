"""
Quality Dividend Mean Reversion
===============================

Selective entry, higher conviction, tight stops

PARAMETERS:
- Entry Window (pre-div): (-5, -2)
- Entry Window (post-div): (0, 3)
- Z-score thresholds: -1.8, -2.2
- Profit Target: 4.0%
- Stop Loss: 1.2%
- Max Holding: 10 days
- Max Positions: 15
- Position Size: 4.0%
"""

from strategy_enhanced_mr import EnhancedDividendMeanReversionStrategy

class OptimizedStrategy(EnhancedDividendMeanReversionStrategy):
    """Optimized variant with custom parameters."""

    def __init__(self, data_manager):
        super().__init__(data_manager)

        # Override with optimized parameters
        self.params.update({
            'pre_div_entry_window': (-5, -2),
            'post_div_entry_window': (0, 3),
            'pre_div_zscore_threshold': -1.8,
            'post_div_zscore_threshold': -2.2,
            'profit_target': 0.04,
            'stop_loss': 0.012,
            'max_holding_days': 10,
            'max_positions': 15,
            'base_position_size': 0.04,
        })
