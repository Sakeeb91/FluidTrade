import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from fluid_dynamics_hft.src.fluid_hft_strategy import FluidHFTStrategy
from fluid_dynamics_hft.src.fluid_market_model import FluidMarketModel

class TestFluidHFTStrategy(unittest.TestCase):
    """Unit tests for the FluidHFTStrategy class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a basic market model
        self.model = FluidMarketModel()
        
        # Generate synthetic test data
        np.random.seed(42)
        n_events = 1000
        
        # Create timestamps
        start_time = pd.Timestamp('2023-01-01 09:30:00')
        times = [start_time + timedelta(seconds=i) for i in range(n_events)]
        
        # Create price data with some patterns
        base_price = 100.0
        price_changes = np.random.normal(0, 0.01, n_events).cumsum()
        prices = base_price + price_changes
        prices = np.maximum(np.round(prices * 100) / 100, 0.01)
        
        # Create order flow data
        signed_volume = np.random.choice([-100, -50, -10, -5, -1, 1, 5, 10, 50, 100], size=n_events)
        
        # Create dataframe
        self.data = pd.DataFrame({
            'DateTime': times,
            'MidPrice': prices,
            'SignedVolume': signed_volume,
            'OrderFlow': signed_volume.cumsum()
        })
        self.data.set_index('DateTime', inplace=True)
        
        # Add to model
        self.model.data = self.data
        
        # Create simple flow, pressure, and vorticity fields for testing
        time_window, price_levels = 20, 10
        self.model.flow_field = np.random.randn(time_window, price_levels)
        self.model.pressure_field = np.random.randn(time_window, price_levels)
        self.model.vorticity_field = np.random.randn(time_window, price_levels)
        
        # Initialize strategy with model
        self.strategy = FluidHFTStrategy(model=self.model)

    def test_initialization(self):
        """Test that the strategy initializes correctly."""
        # Test that model was assigned correctly
        self.assertEqual(self.strategy.model, self.model)
        
        # Test that internal state variables were initialized correctly
        self.assertEqual(len(self.strategy.positions), 0)
        self.assertEqual(len(self.strategy.trades), 0)
        self.assertEqual(len(self.strategy.pnl), 0)
        self.assertEqual(len(self.strategy.cumulative_pnl), 0)
        self.assertEqual(len(self.strategy.holding_periods), 0)
        self.assertEqual(len(self.strategy.trade_types), 0)

    def test_identify_vortex_patterns(self):
        """Test the vortex pattern identification."""
        vortex_signals = self.strategy.identify_vortex_patterns(lookback=10, threshold=0.5)
        
        # Check that signals are returned as a pandas Series
        self.assertIsInstance(vortex_signals, pd.Series)
        
        # Check that signals have the same index as the data
        self.assertEqual(len(vortex_signals), len(self.data))
        self.assertTrue(all(vortex_signals.index == self.data.index))
        
        # Check that signals are either -1, 0, or 1
        self.assertTrue(all(vortex_signals.isin([-1, 0, 1])))
        
        # Check that the pattern type was registered
        self.assertIn('vortex', self.strategy.trade_types)
        self.assertEqual(self.strategy.trade_types['vortex'], 1)

    def test_identify_pressure_gradients(self):
        """Test the pressure gradient identification."""
        pressure_signals = self.strategy.identify_pressure_gradients(lookback=10, threshold=0.5)
        
        # Check that signals are returned as a pandas Series
        self.assertIsInstance(pressure_signals, pd.Series)
        
        # Check that signals have the same index as the data
        self.assertEqual(len(pressure_signals), len(self.data))
        self.assertTrue(all(pressure_signals.index == self.data.index))
        
        # Check that signals are either -1, 0, or 1
        self.assertTrue(all(pressure_signals.isin([-1, 0, 1])))
        
        # Check that the pattern type was registered
        self.assertIn('pressure_gradient', self.strategy.trade_types)
        self.assertEqual(self.strategy.trade_types['pressure_gradient'], 2)

    def test_identify_flow_transitions(self):
        """Test the flow transition pattern identification."""
        transition_signals = self.strategy.identify_flow_transitions(lookback=10, threshold=0.3)
        
        # Check that signals are returned as a pandas Series
        self.assertIsInstance(transition_signals, pd.Series)
        
        # Check that signals have the same index as the data
        self.assertEqual(len(transition_signals), len(self.data))
        self.assertTrue(all(transition_signals.index == self.data.index))
        
        # Check that signals are either -1, 0, or 1
        self.assertTrue(all(transition_signals.isin([-1, 0, 1])))
        
        # Check that the pattern type was registered
        self.assertIn('flow_transition', self.strategy.trade_types)
        self.assertEqual(self.strategy.trade_types['flow_transition'], 3)

    def test_identify_order_flow_imbalance(self):
        """Test the order flow imbalance identification."""
        imbalance_signals = self.strategy.identify_order_flow_imbalance(window=100, threshold=1.0)
        
        # Check that signals are returned as a pandas Series
        self.assertIsInstance(imbalance_signals, pd.Series)
        
        # Check that signals have the same index as the data
        self.assertEqual(len(imbalance_signals), len(self.data))
        self.assertTrue(all(imbalance_signals.index == self.data.index))
        
        # Check that signals are either -1, 0, or 1
        self.assertTrue(all(imbalance_signals.isin([-1, 0, 1])))
        
        # Check that the pattern type was registered
        self.assertIn('order_imbalance', self.strategy.trade_types)
        self.assertEqual(self.strategy.trade_types['order_imbalance'], 4)

    def test_generate_combined_signals(self):
        """Test the combined signal generation."""
        # First generate some individual signals
        self.strategy.identify_vortex_patterns(threshold=0.5)
        self.strategy.identify_pressure_gradients(threshold=0.5)
        
        # Generate combined signals
        combined_signals = self.strategy.generate_combined_signals()
        
        # Check that combined signals are returned as a pandas Series
        self.assertIsInstance(combined_signals, pd.Series)
        
        # Check that combined signals have the same index as the data
        self.assertEqual(len(combined_signals), len(self.data))
        self.assertTrue(all(combined_signals.index == self.data.index))
        
        # Check that combined signals include values from individual signals
        self.assertTrue(all(combined_signals.isin([-1, 0, 1])))

    def test_execute_trades(self):
        """Test trade execution based on signals."""
        # Generate some trading signals
        signals = pd.Series(np.random.choice([-1, 0, 1], size=len(self.data.index)), index=self.data.index)
        
        # Execute trades
        performance = self.strategy.execute_trades(
            signals=signals, 
            capital=100000, 
            position_size=0.1, 
            commission_rate=0.001
        )
        
        # Check that performance metrics were calculated
        self.assertIsInstance(performance, dict)
        self.assertIn('total_return', performance)
        self.assertIn('sharpe_ratio', performance)
        self.assertIn('max_drawdown', performance)
        
        # Check that trades were recorded
        self.assertGreater(len(self.strategy.trades), 0)
        self.assertGreater(len(self.strategy.pnl), 0)
        self.assertGreater(len(self.strategy.cumulative_pnl), 0)

    def test_calculate_max_drawdown(self):
        """Test the maximum drawdown calculation."""
        # Create a PnL curve with a known drawdown
        pnl = [0, 1, 2, 1, 0, -1, 0, 1, 2, 3]
        
        # Calculate max drawdown
        max_dd = self.strategy._calculate_max_drawdown(pnl)
        
        # The max drawdown should be 4 (from 2 to -1)
        self.assertEqual(max_dd, 3)

    def test_walk_forward_analysis(self):
        """Test the walk-forward analysis functionality."""
        # This is a simplified test since full walk-forward analysis is computationally intensive
        # Just verify the method runs without errors and returns results
        try:
            results = self.strategy.walk_forward_analysis(window_size=100, step_size=50)
            self.assertIsInstance(results, dict)
            self.assertIn('performance', results)
            self.assertIn('parameters', results)
        except Exception as e:
            self.fail(f"walk_forward_analysis raised an exception: {e}")

    def test_optimize_parameters(self):
        """Test the parameter optimization functionality."""
        # Define a simple parameter grid
        param_grid = {
            'vortex_threshold': [0.5, 0.7],
            'pressure_threshold': [0.4, 0.6]
        }
        
        # Run optimization
        try:
            results = self.strategy.optimize_parameters(param_grid, metric='total_return')
            self.assertIsInstance(results, dict)
            self.assertIn('best_params', results)
            self.assertIn('best_performance', results)
        except Exception as e:
            self.fail(f"optimize_parameters raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
