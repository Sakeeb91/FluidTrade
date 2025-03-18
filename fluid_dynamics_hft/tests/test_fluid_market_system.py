import unittest
import numpy as np
import pandas as pd
import os
import tempfile
from datetime import datetime, timedelta

from fluid_dynamics_hft.src.fluid_market_system import FluidMarketSystem
from fluid_dynamics_hft.src.fluid_market_model import FluidMarketModel
from fluid_dynamics_hft.src.fluid_hft_strategy import FluidHFTStrategy
from fluid_dynamics_hft.src.pattern_identification import FluidPatternRecognizer

class TestFluidMarketSystem(unittest.TestCase):
    """Unit tests for the FluidMarketSystem class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Initialize system with default parameters
        self.system = FluidMarketSystem()
        
        # Create a custom config for testing
        self.test_config = {
            'time_window': 50,
            'price_levels': 30,
            'smoothing_sigma': 0.8,
            'vortex_threshold': 0.6,
            'pressure_threshold': 0.5,
            'flow_threshold': 0.4,
            'initial_capital': 50000,
            'position_size': 0.05,
            'commission_rate': 0.0005
        }

    def test_initialization(self):
        """Test that the system initializes correctly."""
        # Test default initialization
        self.assertIsNone(self.system.data_path)
        self.assertIsNone(self.system.ticker)
        self.assertIsNone(self.system.date)
        
        # Check default config values
        self.assertEqual(self.system.config['time_window'], 100)
        self.assertEqual(self.system.config['price_levels'], 50)
        self.assertEqual(self.system.config['initial_capital'], 100000)
        
        # Check that components are initialized to None
        self.assertIsNone(self.system.model)
        self.assertIsNone(self.system.pattern_recognizer)
        self.assertIsNone(self.system.strategy)
        
        # Test initialization with custom config
        system = FluidMarketSystem(config=self.test_config)
        self.assertEqual(system.config['time_window'], 50)
        self.assertEqual(system.config['initial_capital'], 50000)
        self.assertEqual(system.config['commission_rate'], 0.0005)

    def test_load_data(self):
        """Test data loading functionality."""
        # Load synthetic data
        data = self.system.load_data(synthetic=True)
        
        # Verify that data was loaded
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        
        # Check that model was initialized
        self.assertIsNotNone(self.system.model)
        self.assertIsInstance(self.system.model, FluidMarketModel)
        
        # Check that data was assigned to the model
        self.assertIs(self.system.data, self.system.model.data)

    def test_calculate_fields(self):
        """Test calculation of fluid dynamics fields."""
        # Load data first
        self.system.load_data(synthetic=True)
        
        # Calculate fields
        model = self.system.calculate_fields()
        
        # Verify that fields were calculated
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.flow_field)
        self.assertIsNotNone(model.pressure_field)
        self.assertIsNotNone(model.vorticity_field)
        
        # Check field dimensions match config
        time_window = self.system.config['time_window']
        price_levels = self.system.config['price_levels']
        
        self.assertEqual(model.flow_field.shape, (time_window, price_levels))
        self.assertEqual(model.pressure_field.shape, (time_window, price_levels))
        self.assertEqual(model.vorticity_field.shape, (time_window, price_levels))

    def test_identify_patterns(self):
        """Test pattern identification functionality."""
        # Load data and calculate fields first
        self.system.load_data(synthetic=True)
        self.system.calculate_fields()
        
        # Identify patterns
        patterns = self.system.identify_patterns()
        
        # Verify that patterns were identified
        self.assertIsNotNone(patterns)
        self.assertIsInstance(patterns, dict)
        
        # Check that pattern recognizer was initialized
        self.assertIsNotNone(self.system.pattern_recognizer)
        self.assertIsInstance(self.system.pattern_recognizer, FluidPatternRecognizer)
        
        # Check that patterns were stored
        self.assertIs(self.system.patterns, patterns)
        
        # Check for common pattern types
        self.assertIn('vortices', patterns)
        self.assertIn('flow_regimes', patterns)
        self.assertIn('pressure_gradients', patterns)

    def test_generate_insights(self):
        """Test market insight generation."""
        # Load data, calculate fields, and identify patterns first
        self.system.load_data(synthetic=True)
        self.system.calculate_fields()
        self.system.identify_patterns()
        
        # Generate insights
        insights = self.system.generate_insights()
        
        # Verify that insights were generated
        self.assertIsNotNone(insights)
        self.assertIsInstance(insights, dict)
        
        # Check that insights were stored
        self.assertIs(self.system.insights, insights)
        
        # Check for common insight categories
        self.assertIn('market_state', insights)
        self.assertIn('vortices', insights)
        self.assertIn('flow_regimes', insights)

    def test_generate_signals(self):
        """Test trading signal generation."""
        # Load data, calculate fields, and identify patterns first
        self.system.load_data(synthetic=True)
        self.system.calculate_fields()
        self.system.identify_patterns()
        
        # Generate signals
        signals = self.system.generate_signals()
        
        # Verify that signals were generated
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, pd.Series)
        
        # Check that signals were stored
        self.assertIs(self.system.signals, signals)
        
        # Check that signals are either -1, 0, or 1
        self.assertTrue(all(signals.isin([-1, 0, 1])))
        
        # Check that strategy was initialized
        self.assertIsNotNone(self.system.strategy)
        self.assertIsInstance(self.system.strategy, FluidHFTStrategy)

    def test_backtest_strategy(self):
        """Test strategy backtesting functionality."""
        # Load data, calculate fields, identify patterns, and generate signals first
        self.system.load_data(synthetic=True)
        self.system.calculate_fields()
        self.system.identify_patterns()
        self.system.generate_signals()
        
        # Run backtest
        performance = self.system.backtest_strategy()
        
        # Verify that performance metrics were calculated
        self.assertIsNotNone(performance)
        self.assertIsInstance(performance, dict)
        
        # Check that performance was stored
        self.assertIs(self.system.performance, performance)
        
        # Check for common performance metrics
        self.assertIn('total_return', performance)
        self.assertIn('sharpe_ratio', performance)
        self.assertIn('max_drawdown', performance)
        self.assertIn('win_rate', performance)

    def test_optimize_strategy(self):
        """Test strategy optimization functionality."""
        # Load data, calculate fields, identify patterns, and generate signals first
        self.system.load_data(synthetic=True)
        self.system.calculate_fields()
        self.system.identify_patterns()
        self.system.generate_signals()
        
        # Define a simple parameter grid for testing
        param_grid = {
            'vortex_threshold': [0.5, 0.7],
            'pressure_threshold': [0.4, 0.6]
        }
        
        # Run optimization
        try:
            optimization_results = self.system.optimize_strategy(param_grid, n_iterations=1)
            
            # Verify that optimization results were returned
            self.assertIsNotNone(optimization_results)
            self.assertIsInstance(optimization_results, dict)
            
            # Check for optimization result properties
            self.assertIn('best_params', optimization_results)
            self.assertIn('best_performance', optimization_results)
            
        except Exception as e:
            self.fail(f"optimize_strategy raised an exception: {e}")

    def test_full_workflow(self):
        """Test the entire system workflow end to end."""
        # Create system with test config
        system = FluidMarketSystem(config=self.test_config)
        
        # Run the full workflow
        try:
            # Load data
            system.load_data(synthetic=True)
            self.assertIsNotNone(system.data)
            
            # Calculate fields
            system.calculate_fields()
            self.assertIsNotNone(system.model.flow_field)
            
            # Identify patterns
            system.identify_patterns()
            self.assertIsNotNone(system.patterns)
            
            # Generate insights
            system.generate_insights()
            self.assertIsNotNone(system.insights)
            
            # Generate signals
            system.generate_signals()
            self.assertIsNotNone(system.signals)
            
            # Backtest strategy
            system.backtest_strategy()
            self.assertIsNotNone(system.performance)
            
            # Check that visualization doesn't cause errors
            visualization_success = system.visualize_results()
            self.assertTrue(visualization_success)
            
        except Exception as e:
            self.fail(f"Full workflow test raised an exception: {e}")

    def test_save_and_load_results(self):
        """Test saving and loading results functionality."""
        # Load data and run system to generate results
        self.system.load_data(synthetic=True)
        self.system.calculate_fields()
        self.system.identify_patterns()
        self.system.generate_signals()
        self.system.backtest_strategy()
        
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save results
            output_path = os.path.join(temp_dir, "test_results.pkl")
            save_success = self.system.save_results(output_path)
            self.assertTrue(save_success)
            self.assertTrue(os.path.exists(output_path))
            
            # Create a new system
            new_system = FluidMarketSystem()
            
            # Load results
            load_success = new_system.load_results(output_path)
            self.assertTrue(load_success)
            
            # Check that data was loaded correctly
            self.assertIsNotNone(new_system.data)
            self.assertIsNotNone(new_system.signals)
            self.assertIsNotNone(new_system.performance)


if __name__ == '__main__':
    unittest.main() 