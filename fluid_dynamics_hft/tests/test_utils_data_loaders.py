import unittest
import numpy as np
import pandas as pd
import os
import tempfile
import gzip
import zipfile
from io import BytesIO
from datetime import datetime

from fluid_dynamics_hft.src.utils.data_loaders import load_lobster_data

class TestDataLoaders(unittest.TestCase):
    """Unit tests for the data_loaders utility module."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_path = self.temp_dir.name
        
        # Test parameters
        self.ticker = "AAPL"
        self.date = "2023-01-01"
        self.level = 5
        
        # Create sample LOBSTER format data
        self.message_data = pd.DataFrame({
            'Time': [34200.123, 34210.456, 34220.789, 34230.012],
            'Type': [1, 2, 1, 4],  # submission, cancellation, submission, execution
            'Order_ID': [1001, 1001, 1002, 1002],
            'Size': [100, 100, 200, 200],
            'Price': [150.25, 150.25, 150.50, 150.50],
            'Direction': [1, 1, -1, -1]  # buy, buy, sell, sell
        })
        
        # Create sample order book data
        self.orderbook_data = pd.DataFrame({
            'AskPrice_1': [150.50, 150.50, 150.75, 150.75],
            'AskSize_1': [200, 200, 300, 300],
            'BidPrice_1': [150.25, 150.25, 150.25, 150.25],
            'BidSize_1': [100, 100, 100, 100],
            'AskPrice_2': [150.75, 150.75, 151.00, 151.00],
            'AskSize_2': [150, 150, 200, 200],
            'BidPrice_2': [150.00, 150.00, 150.00, 150.00],
            'BidSize_2': [50, 50, 50, 50]
        })
        
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        self.temp_dir.cleanup()
        
    def _create_test_files(self, compression=None):
        """Helper method to create test LOBSTER files."""
        # Define file paths
        message_file = f"{self.ticker}_{self.date}_34200000_57600000_message_{self.level}.csv"
        orderbook_file = f"{self.ticker}_{self.date}_34200000_57600000_orderbook_{self.level}.csv"
        
        message_path = os.path.join(self.data_path, message_file)
        orderbook_path = os.path.join(self.data_path, orderbook_file)
        
        # Save files with different compression formats
        if compression is None:
            # Save as regular CSV
            self.message_data.to_csv(message_path, header=False, index=False)
            self.orderbook_data.to_csv(orderbook_path, header=False, index=False)
            return message_path, orderbook_path
            
        elif compression == 'gzip':
            # Save as gzipped CSV
            with gzip.open(message_path + '.gz', 'wb') as f:
                f.write(self.message_data.to_csv(header=False, index=False).encode())
            with gzip.open(orderbook_path + '.gz', 'wb') as f:
                f.write(self.orderbook_data.to_csv(header=False, index=False).encode())
            return message_path + '.gz', orderbook_path + '.gz'
            
        elif compression == 'zip':
            # Save as a zip file containing both CSVs
            zip_path = os.path.join(self.data_path, f"{self.ticker}_{self.date}.zip")
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                zipf.writestr(message_file, self.message_data.to_csv(header=False, index=False))
                zipf.writestr(orderbook_file, self.orderbook_data.to_csv(header=False, index=False))
            return zip_path, None
    
    def test_load_lobster_data_csv(self):
        """Test loading LOBSTER data from CSV files."""
        # Create test files
        self._create_test_files()
        
        # Load data
        message_data, orderbook_data = load_lobster_data(
            ticker=self.ticker,
            date=self.date,
            data_path=self.data_path,
            level=self.level
        )
        
        # Verify data was loaded correctly
        self.assertIsNotNone(message_data)
        self.assertIsNotNone(orderbook_data)
        
        # Check basic properties
        self.assertEqual(len(message_data), len(self.message_data))
        self.assertEqual(len(orderbook_data), len(self.orderbook_data))
        
        # Check column names
        expected_message_cols = ['Time', 'Type', 'Order_ID', 'Size', 'Price', 'Direction', 'DateTime']
        for col in expected_message_cols:
            self.assertIn(col, message_data.columns)
        
        # Check data conversion
        # Verify time to datetime conversion works
        self.assertIsInstance(message_data['DateTime'].iloc[0], pd.Timestamp)
        
        # Verify datetime is based on date parameter
        base_date = pd.Timestamp(self.date)
        self.assertEqual(message_data['DateTime'].iloc[0].date(), base_date.date())
        
        # Check order book columns
        expected_orderbook_cols = [f'AskPrice_{i}' for i in range(1, self.level+1)]
        expected_orderbook_cols += [f'AskSize_{i}' for i in range(1, self.level+1)]
        expected_orderbook_cols += [f'BidPrice_{i}' for i in range(1, self.level+1)]
        expected_orderbook_cols += [f'BidSize_{i}' for i in range(1, self.level+1)]
        
        for col in expected_orderbook_cols:
            self.assertIn(col, orderbook_data.columns)
    
    def test_load_lobster_data_gzip(self):
        """Test loading LOBSTER data from gzipped files."""
        # Create gzipped test files
        self._create_test_files(compression='gzip')
        
        # Load data
        try:
            message_data, orderbook_data = load_lobster_data(
                ticker=self.ticker,
                date=self.date,
                data_path=self.data_path,
                level=self.level
            )
            
            # Verify data was loaded correctly
            self.assertIsNotNone(message_data)
            self.assertIsNotNone(orderbook_data)
            
            # Check basic properties
            self.assertEqual(len(message_data), len(self.message_data))
            self.assertEqual(len(orderbook_data), len(self.orderbook_data))
            
        except Exception as e:
            self.fail(f"Failed to load gzipped LOBSTER data: {e}")
    
    def test_load_lobster_data_zip(self):
        """Test loading LOBSTER data from a zip file."""
        # Create zip test file
        self._create_test_files(compression='zip')
        
        # Load data
        try:
            message_data, orderbook_data = load_lobster_data(
                ticker=self.ticker,
                date=self.date,
                data_path=self.data_path,
                level=self.level
            )
            
            # Verify data was loaded correctly
            self.assertIsNotNone(message_data)
            self.assertIsNotNone(orderbook_data)
            
            # Check basic properties
            self.assertEqual(len(message_data), len(self.message_data))
            self.assertEqual(len(orderbook_data), len(self.orderbook_data))
            
        except Exception as e:
            self.fail(f"Failed to load zipped LOBSTER data: {e}")
    
    def test_load_lobster_data_missing_files(self):
        """Test handling of missing data files."""
        # Try to load data without creating any files
        try:
            message_data, orderbook_data = load_lobster_data(
                ticker=self.ticker,
                date=self.date,
                data_path=self.data_path,
                level=self.level
            )
            
            # Should raise an exception
            self.fail("Should have raised an exception for missing files")
            
        except Exception:
            # Expected behavior
            pass
    
    def test_load_lobster_data_no_convert_time(self):
        """Test loading data without time conversion."""
        # Create test files
        self._create_test_files()
        
        # Load data without time conversion
        message_data, orderbook_data = load_lobster_data(
            ticker=self.ticker,
            date=self.date,
            data_path=self.data_path,
            level=self.level,
            convert_time=False
        )
        
        # Verify DateTime column was not added
        self.assertNotIn('DateTime', message_data.columns)
        
        # Original Time column should still be present
        self.assertIn('Time', message_data.columns)


if __name__ == '__main__':
    unittest.main() 