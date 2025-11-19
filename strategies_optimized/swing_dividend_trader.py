"""
Swing Dividend Trader
=====================

Longer holds, larger targets, room to breathe

PARAMETERS:
- Entry Window (pre-div): (-10, -3)
- Entry Window (post-div): (0, 6)
- Z-score thresholds: -1.0, -1.5
- Profit Target: 5.0%
- Stop Loss: 2.5%
- Max Holding: 15 days
- Max Positions: 18
- Position Size: 4.0%
"""

from strategy_enhanced_mr import EnhancedDividendMeanReversionStrategy

class OptimizedStrategy(EnhancedDividendMeanReversionStrategy):
    """Optimized variant with custom parameters."""

    def __init__(self, data_manager):
        super().__init__(data_manager)

        # Override with optimized parameters
        self.params.update({
            'pre_div_entry_window': (-10, -3),
            'post_div_entry_window': (0, 6),
            'pre_div_zscore_threshold': -1.0,
            'post_div_zscore_threshold': -1.5,
            'profit_target': 0.05,
            'stop_loss': 0.025,
            'max_holding_days': 15,
            'max_positions': 18,
            'base_position_size': 0.04,
        })
