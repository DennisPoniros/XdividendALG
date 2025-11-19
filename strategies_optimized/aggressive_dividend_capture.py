"""
Aggressive Dividend Capture
===========================

Wide entry window, high frequency, moderate stops

PARAMETERS:
- Entry Window (pre-div): (-14, -1)
- Entry Window (post-div): (0, 5)
- Z-score thresholds: -0.8, -1.2
- Profit Target: 2.5%
- Stop Loss: 1.5%
- Max Holding: 12 days
- Max Positions: 25
- Position Size: 3.0%
"""

from strategy_enhanced_mr import EnhancedDividendMeanReversionStrategy

class OptimizedStrategy(EnhancedDividendMeanReversionStrategy):
    """Optimized variant with custom parameters."""

    def __init__(self, data_manager):
        super().__init__(data_manager)

        # Override with optimized parameters
        self.params.update({
            'pre_div_entry_window': (-14, -1),
            'post_div_entry_window': (0, 5),
            'pre_div_zscore_threshold': -0.8,
            'post_div_zscore_threshold': -1.2,
            'profit_target': 0.025,
            'stop_loss': 0.015,
            'max_holding_days': 12,
            'max_positions': 25,
            'base_position_size': 0.03,
        })
