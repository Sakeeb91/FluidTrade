"""
Data loading utilities for the fluid dynamics HFT system.

This module provides utilities for loading and preprocessing market data from various sources,
including LOBSTER format, synthetic data generation, and data validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import logging
import os
import pickle
from pathlib import Path


class DataLoader:
    """
    Base class for data loading utilities.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the data loader.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from specified source.
        
        Args:
            source: Data source identifier
            **kwargs: Additional arguments for data loading
            
        Returns:
            DataFrame with loaded data
        """
        raise NotImplementedError("Subclasses must implement load_data method")
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate loaded data.
        
        Args:
            data: DataFrame to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        if data.empty:
            self.logger.error("Data is empty")
            return False
        
        required_columns = ['Time', 'Price', 'Size']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            return False
        
        # Check for null values
        if data[required_columns].isnull().any().any():
            self.logger.warning("Data contains null values")
        
        # Check for reasonable price values
        if (data['Price'] <= 0).any():
            self.logger.error("Data contains non-positive prices")
            return False
        
        return True


class SyntheticDataGenerator(DataLoader):
    """
    Generator for synthetic market data with realistic properties.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.default_params = {
            'base_price': 100.0,
            'volatility': 0.02,
            'drift': 0.0001,
            'tick_size': 0.01,
            'mean_reversion_speed': 0.1,
            'jump_probability': 0.001,
            'jump_size_std': 0.05
        }
        
    def load_data(self, n_points: int = 10000, **kwargs) -> pd.DataFrame:
        """
        Generate synthetic market data.
        
        Args:
            n_points: Number of data points to generate
            **kwargs: Override default parameters
            
        Returns:
            DataFrame with synthetic market data
        """
        params = {**self.default_params, **kwargs}
        
        # Generate timestamps
        start_time = datetime.now() - timedelta(hours=1)
        timestamps = [start_time + timedelta(seconds=i*0.1) for i in range(n_points)]
        
        # Generate price series with mean reversion and jumps
        prices = self._generate_price_series(n_points, params)
        
        # Generate volume series
        volumes = self._generate_volume_series(n_points, params)
        
        # Generate signed volume (order flow)
        signed_volumes = self._generate_signed_volume_series(n_points, params)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Time': timestamps,
            'Price': prices,
            'Size': volumes,
            'Type': np.ones(n_points),  # Default to type 1
            'Order_ID': np.arange(1, n_points + 1),
            'Direction': np.sign(signed_volumes)
        })
        
        return data
    
    def _generate_price_series(self, n_points: int, params: Dict) -> np.ndarray:
        """Generate realistic price series with mean reversion and jumps."""
        prices = np.zeros(n_points)
        prices[0] = params['base_price']
        
        for i in range(1, n_points):
            # Mean reversion component
            mean_reversion = -params['mean_reversion_speed'] * (prices[i-1] - params['base_price'])
            
            # Drift component
            drift = params['drift']
            
            # Diffusion component
            diffusion = np.random.normal(0, params['volatility'])
            
            # Jump component
            jump = 0
            if np.random.random() < params['jump_probability']:
                jump = np.random.normal(0, params['jump_size_std'])
            
            # Update price
            price_change = mean_reversion + drift + diffusion + jump
            prices[i] = max(prices[i-1] + price_change, params['tick_size'])
        
        # Round to tick size
        prices = np.round(prices / params['tick_size']) * params['tick_size']
        
        return prices
    
    def _generate_volume_series(self, n_points: int, params: Dict) -> np.ndarray:
        """Generate realistic volume series."""
        # Base volume with some time-of-day effects
        base_volumes = np.random.exponential(scale=500, size=n_points)
        
        # Add time-of-day effects (higher volume at open/close)
        time_effect = np.sin(np.linspace(0, 2*np.pi, n_points)) * 0.3 + 1
        volumes = base_volumes * time_effect
        
        # Round to meaningful sizes
        volumes = np.round(volumes / 100) * 100
        volumes = np.maximum(volumes, 100)  # Minimum 100 shares
        
        return volumes.astype(int)
    
    def _generate_signed_volume_series(self, n_points: int, params: Dict) -> np.ndarray:
        """Generate signed volume series (order flow)."""
        # Generate autocorrelated signed volume
        signed_volumes = np.zeros(n_points)
        signed_volumes[0] = np.random.normal(0, 1000)
        
        persistence = 0.1  # Autocorrelation parameter
        
        for i in range(1, n_points):
            # Autocorrelated component
            autocorr = persistence * signed_volumes[i-1]
            
            # Random shock
            shock = np.random.normal(0, 1000)
            
            signed_volumes[i] = autocorr + shock
        
        return signed_volumes


def load_market_data(ticker: str = "SYNTHETIC", date: str = None, data_source: str = 'synthetic', 
                    **kwargs) -> pd.DataFrame:
    """
    Convenience function to load market data from various sources.
    
    Args:
        ticker: Stock ticker
        date: Date in YYYY-MM-DD format
        data_source: Data source ('synthetic')
        **kwargs: Additional arguments
        
    Returns:
        DataFrame with market data
    """
    if date is None:
        date = datetime.now().strftime('%Y-%m-%d')
    
    # Load data based on source
    if data_source == 'synthetic':
        loader = SyntheticDataGenerator()
        data = loader.load_data(**kwargs)
    else:
        raise ValueError(f"Unknown data source: {data_source}")
    
    return data