"""
FluidMarketModel: A physics-based approach to modeling market dynamics using fluid dynamics principles.

This module implements a market model that treats order flow as fluid flow, price levels as spatial
coordinates, and market pressure as a scalar field. The model uses computational fluid dynamics
techniques to analyze market behavior and predict price movements.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple, Any
import logging
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks


class FluidMarketModel:
    """
    A market model based on fluid dynamics principles.
    
    This model treats the order book as a fluid system where:
    - Order flow represents fluid velocity
    - Price levels represent spatial coordinates
    - Market pressure represents scalar pressure field
    - Volume represents fluid density
    """
    
    def __init__(self, data_path: Optional[str] = None, ticker: Optional[str] = None, 
                 date: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize the FluidMarketModel.
        
        Args:
            data_path: Path to market data files
            ticker: Stock ticker symbol
            date: Date for analysis
            config: Configuration parameters
        """
        self.data_path = data_path
        self.ticker = ticker
        self.date = date
        self.config = config or {}
        
        # Data storage
        self.data: Optional[pd.DataFrame] = None
        self.flow_field: Optional[np.ndarray] = None
        self.pressure_field: Optional[np.ndarray] = None
        self.vorticity_field: Optional[np.ndarray] = None
        self.patterns: Dict[str, Any] = {}
        
        # Model parameters
        self.viscosity = self.config.get('viscosity', 0.01)
        self.density = self.config.get('density', 1.0)
        self.time_step = self.config.get('time_step', 0.01)
        
        # Grid parameters
        self.time_window = self.config.get('time_window', 100)
        self.price_levels = self.config.get('price_levels', 50)
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def load_lobster_data(self, synthetic: bool = True) -> pd.DataFrame:
        """
        Load market data (LOBSTER format or synthetic).
        
        Args:
            synthetic: Whether to generate synthetic data
            
        Returns:
            DataFrame with market data
        """
        if synthetic or not self.data_path:
            return self._generate_synthetic_data()
        else:
            return self._load_real_data()
    
    def _generate_synthetic_data(self) -> pd.DataFrame:
        """Generate synthetic market data for testing."""
        np.random.seed(42)
        n_events = 10000
        
        # Create timestamps
        start_time = pd.Timestamp('2023-01-01 09:30:00')
        times = [start_time + timedelta(milliseconds=i*100) for i in range(n_events)]
        
        # Generate realistic order flow patterns
        base_price = 100.0
        
        # Price evolution with mean reversion
        price_changes = []
        current_price = base_price
        for i in range(n_events):
            # Mean reversion force
            mean_reversion = -0.001 * (current_price - base_price)
            # Random shock
            shock = np.random.normal(0, 0.005)
            # Momentum effect
            momentum = 0.1 * (price_changes[-1] if price_changes else 0)
            
            change = mean_reversion + shock + momentum
            price_changes.append(change)
            current_price += change
        
        prices = base_price + np.cumsum(price_changes)
        prices = np.maximum(np.round(prices * 100) / 100, 0.01)
        
        # Generate order types and sizes
        order_types = np.random.choice([1, 2, 3, 4, 5], size=n_events, 
                                     p=[0.4, 0.3, 0.1, 0.1, 0.1])
        order_ids = np.arange(1, n_events + 1)
        sizes = np.random.choice([1, 5, 10, 50, 100], size=n_events,
                               p=[0.5, 0.3, 0.15, 0.04, 0.01])
        directions = np.random.choice([-1, 1], size=n_events)
        
        # Create DataFrame
        data = pd.DataFrame({
            'Time': times,
            'Type': order_types,
            'Order_ID': order_ids,
            'Size': sizes,
            'Price': prices,
            'Direction': directions
        })
        
        self.data = data
        return data
    
    def _load_real_data(self) -> pd.DataFrame:
        """Load real market data from file."""
        try:
            # Placeholder for real data loading
            # In practice, this would load LOBSTER or similar format
            data = pd.read_csv(self.data_path)
            self.data = data
            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return self._generate_synthetic_data()
    
    def compute_flow_field(self) -> np.ndarray:
        """
        Compute the flow field from order flow data.
        
        The flow field represents the velocity of liquidity at each price level
        and time point.
        
        Returns:
            2D array representing flow field (time x price_levels)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_lobster_data() first.")
        
        # Create price grid
        min_price = self.data['Price'].min()
        max_price = self.data['Price'].max()
        price_grid = np.linspace(min_price, max_price, self.price_levels)
        
        # Create time grid
        time_grid = pd.date_range(start=self.data['Time'].min(), 
                                 end=self.data['Time'].max(), 
                                 periods=self.time_window)
        
        # Initialize flow field
        flow_field = np.zeros((self.time_window, self.price_levels))
        
        # Compute signed volume flow
        for i, time_point in enumerate(time_grid[:-1]):
            # Get orders in this time window
            time_mask = (self.data['Time'] >= time_point) & \
                       (self.data['Time'] < time_grid[i + 1])
            window_data = self.data[time_mask]
            
            if len(window_data) == 0:
                continue
                
            # Compute flow at each price level
            for j, price_level in enumerate(price_grid):
                # Find orders near this price level
                price_tolerance = (max_price - min_price) / (2 * self.price_levels)
                price_mask = np.abs(window_data['Price'] - price_level) <= price_tolerance
                relevant_orders = window_data[price_mask]
                
                if len(relevant_orders) > 0:
                    # Compute net flow (signed volume)
                    net_flow = (relevant_orders['Size'] * relevant_orders['Direction']).sum()
                    flow_field[i, j] = net_flow
        
        # Apply spatial smoothing
        flow_field = gaussian_filter(flow_field, sigma=1.0)
        
        self.flow_field = flow_field
        return flow_field
    
    def compute_pressure_field(self) -> np.ndarray:
        """
        Compute the pressure field from price and volume data.
        
        The pressure field represents market pressure at each price level.
        High pressure indicates strong buying/selling interest.
        
        Returns:
            2D array representing pressure field (time x price_levels)
        """
        if self.flow_field is None:
            self.compute_flow_field()
        
        # Pressure is related to the divergence of flow field
        #  · u = u/x + v/y
        pressure_field = np.zeros_like(self.flow_field)
        
        # Compute spatial derivatives
        flow_dx = np.gradient(self.flow_field, axis=1)  # Price direction
        flow_dt = np.gradient(self.flow_field, axis=0)  # Time direction
        
        # Pressure relates to flow divergence and acceleration
        pressure_field = -flow_dx - 0.1 * flow_dt
        
        # Add volume-based pressure
        if self.data is not None:
            min_price = self.data['Price'].min()
            max_price = self.data['Price'].max()
            price_grid = np.linspace(min_price, max_price, self.price_levels)
            time_grid = pd.date_range(start=self.data['Time'].min(), 
                                     end=self.data['Time'].max(), 
                                     periods=self.time_window)
            
            for i, time_point in enumerate(time_grid[:-1]):
                time_mask = (self.data['Time'] >= time_point) & \
                           (self.data['Time'] < time_grid[i + 1])
                window_data = self.data[time_mask]
                
                for j, price_level in enumerate(price_grid):
                    price_tolerance = (max_price - min_price) / (2 * self.price_levels)
                    price_mask = np.abs(window_data['Price'] - price_level) <= price_tolerance
                    relevant_orders = window_data[price_mask]
                    
                    if len(relevant_orders) > 0:
                        # Volume-based pressure
                        total_volume = relevant_orders['Size'].sum()
                        pressure_field[i, j] += 0.01 * total_volume
        
        # Apply temporal smoothing
        pressure_field = gaussian_filter(pressure_field, sigma=(2.0, 1.0))
        
        self.pressure_field = pressure_field
        return pressure_field
    
    def compute_vorticity_field(self) -> np.ndarray:
        """
        Compute the vorticity field from the flow field.
        
        Vorticity measures the local rotation of the flow field, which can
        indicate market regimes and potential reversals.
        
        Returns:
            2D array representing vorticity field (time x price_levels)
        """
        if self.flow_field is None:
            self.compute_flow_field()
        
        # Vorticity is curl of velocity field
        # For 2D: É = v/x - u/y
        # We treat the flow field as u-component and estimate v from continuity
        
        vorticity_field = np.zeros_like(self.flow_field)
        
        # Compute spatial derivatives
        flow_dx = np.gradient(self.flow_field, axis=1)  # u/x
        flow_dy = np.gradient(self.flow_field, axis=0)  # u/t (proxy for v/y)
        
        # Approximate vorticity
        vorticity_field = flow_dx - 0.1 * flow_dy
        
        # Apply smoothing
        vorticity_field = gaussian_filter(vorticity_field, sigma=1.5)
        
        self.vorticity_field = vorticity_field
        return vorticity_field
    
    def detect_regime_transitions(self) -> Dict[str, List[int]]:
        """
        Detect market regime transitions using fluid dynamics principles.
        
        Returns:
            Dictionary mapping regime types to time indices
        """
        if self.pressure_field is None:
            self.compute_pressure_field()
        if self.vorticity_field is None:
            self.compute_vorticity_field()
        
        transitions = {
            'high_pressure': [],
            'low_pressure': [],
            'high_vorticity': [],
            'low_vorticity': [],
            'flow_reversal': []
        }
        
        # Detect high/low pressure regimes
        pressure_mean = np.mean(self.pressure_field, axis=1)
        pressure_std = np.std(self.pressure_field, axis=1)
        
        high_pressure_threshold = np.percentile(pressure_mean, 80)
        low_pressure_threshold = np.percentile(pressure_mean, 20)
        
        high_pressure_indices = np.where(pressure_mean > high_pressure_threshold)[0]
        low_pressure_indices = np.where(pressure_mean < low_pressure_threshold)[0]
        
        transitions['high_pressure'] = high_pressure_indices.tolist()
        transitions['low_pressure'] = low_pressure_indices.tolist()
        
        # Detect high/low vorticity regimes
        vorticity_mean = np.mean(np.abs(self.vorticity_field), axis=1)
        high_vorticity_threshold = np.percentile(vorticity_mean, 80)
        low_vorticity_threshold = np.percentile(vorticity_mean, 20)
        
        high_vorticity_indices = np.where(vorticity_mean > high_vorticity_threshold)[0]
        low_vorticity_indices = np.where(vorticity_mean < low_vorticity_threshold)[0]
        
        transitions['high_vorticity'] = high_vorticity_indices.tolist()
        transitions['low_vorticity'] = low_vorticity_indices.tolist()
        
        # Detect flow reversals
        flow_mean = np.mean(self.flow_field, axis=1)
        flow_changes = np.diff(flow_mean)
        reversal_indices = []
        
        for i in range(1, len(flow_changes)):
            if flow_changes[i-1] * flow_changes[i] < 0:  # Sign change
                if abs(flow_changes[i]) > np.std(flow_changes):
                    reversal_indices.append(i)
        
        transitions['flow_reversal'] = reversal_indices
        
        return transitions
    
    def get_market_state(self, time_index: int) -> Dict[str, float]:
        """
        Get the current market state at a specific time index.
        
        Args:
            time_index: Index in the time dimension
            
        Returns:
            Dictionary with market state metrics
        """
        if any(field is None for field in [self.flow_field, self.pressure_field, self.vorticity_field]):
            self.compute_flow_field()
            self.compute_pressure_field()
            self.compute_vorticity_field()
        
        if time_index >= self.time_window:
            raise ValueError(f"Time index {time_index} exceeds window size {self.time_window}")
        
        state = {
            'flow_intensity': np.mean(np.abs(self.flow_field[time_index, :])),
            'pressure_level': np.mean(self.pressure_field[time_index, :]),
            'vorticity_magnitude': np.mean(np.abs(self.vorticity_field[time_index, :])),
            'flow_concentration': np.std(self.flow_field[time_index, :]),
            'pressure_gradient': np.max(np.abs(np.gradient(self.pressure_field[time_index, :]))),
            'flow_direction': np.sign(np.mean(self.flow_field[time_index, :]))
        }
        
        return state
    
    def predict_price_movement(self, time_index: int, horizon: int = 5) -> Dict[str, float]:
        """
        Predict price movement based on fluid dynamics.
        
        Args:
            time_index: Current time index
            horizon: Prediction horizon
            
        Returns:
            Dictionary with prediction metrics
        """
        if time_index + horizon >= self.time_window:
            horizon = self.time_window - time_index - 1
        
        current_state = self.get_market_state(time_index)
        
        # Predict based on pressure gradients and flow patterns
        pressure_trend = np.mean(self.pressure_field[time_index:time_index+horizon, :], axis=1)
        flow_trend = np.mean(self.flow_field[time_index:time_index+horizon, :], axis=1)
        
        # Simple prediction model
        pressure_momentum = np.gradient(pressure_trend).mean() if len(pressure_trend) > 1 else 0
        flow_momentum = np.gradient(flow_trend).mean() if len(flow_trend) > 1 else 0
        
        prediction = {
            'price_direction': np.sign(flow_momentum + 0.5 * pressure_momentum),
            'confidence': min(1.0, abs(flow_momentum) + abs(pressure_momentum)),
            'volatility_forecast': current_state['vorticity_magnitude'],
            'pressure_momentum': pressure_momentum,
            'flow_momentum': flow_momentum
        }
        
        return prediction