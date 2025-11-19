"""
Balanced Alpha Generator
========================

Balanced approach, consistent returns

PARAMETERS:
- Entry Window (pre-div): (-8, -2)
- Entry Window (post-div): (0, 4)
- Z-score thresholds: -1.2, -1.6
- Profit Target: 3.0%
- Stop Loss: 1.8%
- Max Holding: 11 days
- Max Positions: 20
- Position Size: 3.5%
"""

from strategy_enhanced_mr import EnhancedDividendMeanReversionStrategy

class OptimizedStrategy(EnhancedDividendMeanReversionStrategy):
    """Optimized variant with custom parameters."""

    def __init__(self, data_manager):
        super().__init__(data_manager)

        # Override with optimized parameters
        self.params.update({
            'pre_div_entry_window': (-8, -2),
            'post_div_entry_window': (0, 4),
            'pre_div_zscore_threshold': -1.2,
            'post_div_zscore_threshold': -1.6,
            'profit_target': 0.03,
            'stop_loss': 0.018,
            'max_holding_days': 11,
            'max_positions': 20,
            'base_position_size': 0.035,
        })
