import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import tempfile

from fluid_dynamics_hft.src.fluid_market_model import FluidMarketModel

class TestFluidMarketModel(unittest.TestCase):
    """Unit tests for the FluidMarketModel class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Initialize model with default parameters
        self.model = FluidMarketModel()

    def test_initialization(self):
        """Test that the model initializes correctly."""
        # Test default initialization
        self.assertIsNone(self.model.data_path)
        self.assertIsNone(self.model.ticker)
        self.assertIsNone(self.model.date)
        self.assertIsNone(self.model.data)
        self.assertIsNone(self.model.flow_field)
        self.assertIsNone(self.model.pressure_field)
        self.assertIsNone(self.model.vorticity_field)
        self.assertEqual(self.model.patterns, {})
        
        # Test initialization with parameters
        test_path = "/path/to/data"
        test_ticker = "AAPL"
        test_date = "2023-01-01"
        model = FluidMarketModel(data_path=test_path, ticker=test_ticker, date=test_date)
        
        self.assertEqual(model.data_path, test_path)
        self.assertEqual(model.ticker, test_ticker)
        self.assertEqual(model.date, test_date)

    def test_load_lobster_data_synthetic(self):
        """Test loading synthetic lobster data."""
        # Call function to generate synthetic data
        data = self.model.load_lobster_data()
        
        # Verify that data was generated and has the expected structure
        self.assertIsNotNone(data)
        self.assertIsInstance(data, pd.DataFrame)
        
        # Check that required columns are present
        expected_columns = ['Time', 'Type', 'Order_ID', 'Size', 'Price', 'Direction']
        for col in expected_columns:
            self.assertIn(col, data.columns)
        
        # Check that derived columns were calculated correctly
        self.assertIn('DateTime', data.columns)
        self.assertIn('PriceLevel', data.columns)
        self.assertIn('SignedVolume', data.columns)
        self.assertIn('OrderFlow', data.columns)
        self.assertIn('MidPrice', data.columns)
        self.assertIn('OrderImbalance', data.columns)

    def test_preprocess_data(self):
        """Test data preprocessing functionality."""
        # Generate synthetic data first
        self.model.load_lobster_data()
        
        # Store original columns for comparison
        original_columns = set(self.model.data.columns)
        
        # Call preprocess_data again
        self.model.preprocess_data()
        
        # Verify preprocessing results
        self.assertIn('DateTime', self.model.data.columns)
        self.assertIn('PriceLevel', self.model.data.columns)
        self.assertIn('SignedVolume', self.model.data.columns)
        self.assertIn('OrderFlow', self.model.data.columns)
        self.assertIn('MidPrice', self.model.data.columns)
        
        # Check that SignedVolume is calculated correctly
        sample_row = self.model.data.iloc[0]
        self.assertEqual(sample_row['SignedVolume'], sample_row['Size'] * sample_row['Direction'])

    def test_compute_mid_prices(self):
        """Test the mid-price computation."""
        # Generate synthetic data first
        self.model.load_lobster_data()
        
        # Call the method to test
        self.model.compute_mid_prices()
        
        # Verify results
        self.assertIn('BestBid', self.model.data.columns)
        self.assertIn('BestAsk', self.model.data.columns)
        self.assertIn('MidPrice', self.model.data.columns)
        self.assertIn('MidPriceChange', self.model.data.columns)
        
        # Check that mid price is between bid and ask where values are available
        for i, row in self.model.data.iterrows():
            if not np.isnan(row['BestBid']) and not np.isnan(row['BestAsk']):
                self.assertLessEqual(row['BestBid'], row['MidPrice'])
                self.assertGreaterEqual(row['BestAsk'], row['MidPrice'])

    def test_compute_order_book_pressure(self):
        """Test the order book pressure computation."""
        # Generate synthetic data first
        self.model.load_lobster_data()
        
        # Call the method to test
        self.model.compute_order_book_pressure()
        
        # Verify results
        self.assertIn('OrderImbalance', self.model.data.columns)
        self.assertIn('OrderBookPressure', self.model.data.columns)
        
        # Check range of OrderImbalance (should be between -1 and 1)
        max_imbalance = self.model.data['OrderImbalance'].max()
        min_imbalance = self.model.data['OrderImbalance'].min()
        
        self.assertLessEqual(max_imbalance, 1.0)
        self.assertGreaterEqual(min_imbalance, -1.0)

    def test_calculate_flow_field(self):
        """Test the flow field calculation."""
        # Generate synthetic data first
        self.model.load_lobster_data()
        
        # Define test parameters
        time_window = 50
        price_levels = 30
        
        # Calculate flow field
        self.model.calculate_flow_field(time_window=time_window, price_levels=price_levels)
        
        # Verify flow field dimensions
        self.assertIsNotNone(self.model.flow_field)
        self.assertEqual(self.model.flow_field.shape, (time_window, price_levels))
        
        # Check that values are normalized (between -1 and 1)
        self.assertLessEqual(np.max(self.model.flow_field), 1.0)
        self.assertGreaterEqual(np.min(self.model.flow_field), -1.0)

    def test_calculate_pressure_field(self):
        """Test the pressure field calculation."""
        # Generate synthetic data first
        self.model.load_lobster_data()
        
        # Need to calculate flow field first
        time_window = 50
        price_levels = 30
        self.model.calculate_flow_field(time_window=time_window, price_levels=price_levels)
        
        # Calculate pressure field
        self.model.calculate_pressure_field()
        
        # Verify pressure field dimensions
        self.assertIsNotNone(self.model.pressure_field)
        self.assertEqual(self.model.pressure_field.shape, (time_window, price_levels))

    def test_calculate_vorticity_field(self):
        """Test the vorticity field calculation."""
        # Generate synthetic data first
        self.model.load_lobster_data()
        
        # Need to calculate flow field first
        time_window = 50
        price_levels = 30
        self.model.calculate_flow_field(time_window=time_window, price_levels=price_levels)
        
        # Calculate vorticity field
        self.model.calculate_vorticity_field()
        
        # Verify vorticity field dimensions
        self.assertIsNotNone(self.model.vorticity_field)
        self.assertEqual(self.model.vorticity_field.shape, (time_window, price_levels))

    def test_identify_flow_patterns(self):
        """Test the flow pattern identification."""
        # Generate synthetic data and calculate fields
        self.model.load_lobster_data()
        
        # Calculate required fields
        time_window = 50
        price_levels = 30
        self.model.calculate_flow_field(time_window=time_window, price_levels=price_levels)
        self.model.calculate_pressure_field()
        self.model.calculate_vorticity_field()
        
        # Identify patterns
        patterns = self.model.identify_flow_patterns()
        
        # Verify patterns were identified
        self.assertIsNotNone(patterns)
        self.assertIsInstance(patterns, dict)
        
        # Check for expected pattern types
        expected_pattern_types = ['vortices', 'laminar', 'turbulent', 'pressure_gradients']
        for pattern_type in expected_pattern_types:
            self.assertIn(pattern_type, patterns)

    def test_identify_trading_signals(self):
        """Test the trading signal identification."""
        # Generate synthetic data and calculate fields
        self.model.load_lobster_data()
        
        # Calculate required fields
        time_window = 50
        price_levels = 30
        self.model.calculate_flow_field(time_window=time_window, price_levels=price_levels)
        self.model.calculate_pressure_field()
        self.model.calculate_vorticity_field()
        
        # Identify patterns first
        self.model.identify_flow_patterns()
        
        # Identify trading signals
        signals = self.model.identify_trading_signals()
        
        # Verify signals were identified
        self.assertIsNotNone(signals)
        self.assertIsInstance(signals, pd.Series)
        
        # Check that signals have the same index as the data
        self.assertEqual(len(signals), len(self.model.data))
        
        # Check that signals are either -1, 0, or 1
        self.assertTrue(all(signals.isin([-1, 0, 1])))

    def test_model_workflow(self):
        """Test the entire workflow from data loading to signal generation."""
        # Initialize a new model
        model = FluidMarketModel()
        
        # Load synthetic data
        model.load_lobster_data()
        self.assertIsNotNone(model.data)
        
        # Calculate fields
        model.calculate_flow_field()
        self.assertIsNotNone(model.flow_field)
        
        model.calculate_pressure_field()
        self.assertIsNotNone(model.pressure_field)
        
        model.calculate_vorticity_field()
        self.assertIsNotNone(model.vorticity_field)
        
        # Identify patterns
        patterns = model.identify_flow_patterns()
        self.assertIsNotNone(patterns)
        
        # Generate signals
        signals = model.identify_trading_signals()
        self.assertIsNotNone(signals)
        
        # Test backtest functionality
        try:
            backtest_results = model.backtest_strategy()
            self.assertIsInstance(backtest_results, dict)
            self.assertIn('total_return', backtest_results)
            self.assertIn('sharpe_ratio', backtest_results)
        except Exception as e:
            self.fail(f"backtest_strategy raised an exception: {e}")


if __name__ == '__main__':
    unittest.main()
