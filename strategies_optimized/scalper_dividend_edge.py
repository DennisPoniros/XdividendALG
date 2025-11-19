"""
Scalper Dividend Edge
=====================

Quick in/out, high frequency, small gains

PARAMETERS:
- Entry Window (pre-div): (-6, -1)
- Entry Window (post-div): (0, 2)
- Z-score thresholds: -0.5, -1.0
- Profit Target: 1.5%
- Stop Loss: 1.0%
- Max Holding: 7 days
- Max Positions: 30
- Position Size: 2.5%
"""

from strategy_enhanced_mr import EnhancedDividendMeanReversionStrategy

class OptimizedStrategy(EnhancedDividendMeanReversionStrategy):
    """Optimized variant with custom parameters."""

    def __init__(self, data_manager):
        super().__init__(data_manager)

        # Override with optimized parameters
        self.params.update({
            'pre_div_entry_window': (-6, -1),
            'post_div_entry_window': (0, 2),
            'pre_div_zscore_threshold': -0.5,
            'post_div_zscore_threshold': -1.0,
            'profit_target': 0.015,
            'stop_loss': 0.01,
            'max_holding_days': 7,
            'max_positions': 30,
            'base_position_size': 0.025,
        })
