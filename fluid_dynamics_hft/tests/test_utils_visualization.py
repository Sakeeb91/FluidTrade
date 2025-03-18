import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import io
import os

from fluid_dynamics_hft.src.utils.visualization import (
    plot_flow_field, plot_vorticity_field, plot_pressure_field, 
    plot_pattern_overlay
)

class TestVisualization(unittest.TestCase):
    """Unit tests for the visualization utility module."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create synthetic test fields for visualization
        np.random.seed(42)
        self.time_steps, self.price_levels = 30, 20
        
        # 1. Create flow field
        self.flow_field = np.zeros((self.time_steps, self.price_levels))
        
        # Add a laminar flow region
        for i in range(5, 15):
            for j in range(5, 15):
                self.flow_field[i, j] = 0.8
        
        # Add some random noise
        self.flow_field += np.random.normal(0, 0.1, (self.time_steps, self.price_levels))
        
        # 2. Create a vorticity field
        self.vorticity_field = np.zeros((self.time_steps, self.price_levels))
        
        # Add a positive vortex
        center_i, center_j = 10, 10
        radius = 3
        for i in range(self.time_steps):
            for j in range(self.price_levels):
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if dist < radius:
                    self.vorticity_field[i, j] = 0.9 * (1 - dist/radius)
        
        # Add a negative vortex
        center_i, center_j = 20, 15
        radius = 2
        for i in range(self.time_steps):
            for j in range(self.price_levels):
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if dist < radius:
                    self.vorticity_field[i, j] = -0.8 * (1 - dist/radius)
        
        # Add some random noise
        self.vorticity_field += np.random.normal(0, 0.05, (self.time_steps, self.price_levels))
        
        # 3. Create a pressure field
        self.pressure_field = np.zeros((self.time_steps, self.price_levels))
        
        # Add a high pressure area
        center_i, center_j = 15, 8
        radius = 4
        for i in range(self.time_steps):
            for j in range(self.price_levels):
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if dist < radius:
                    self.pressure_field[i, j] = 0.8 * (1 - dist/radius)
        
        # Add a low pressure area
        center_i, center_j = 8, 15
        radius = 3
        for i in range(self.time_steps):
            for j in range(self.price_levels):
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if dist < radius:
                    self.pressure_field[i, j] = -0.7 * (1 - dist/radius)
        
        # Add some random noise
        self.pressure_field += np.random.normal(0, 0.05, (self.time_steps, self.price_levels))
        
        # 4. Create sample patterns dictionary
        self.patterns = {
            'vortices': [
                {
                    'center': (10, 10),
                    'size': 28,
                    'strength': 0.85,
                    'direction': 'counterclockwise',
                    'sign': 1
                },
                {
                    'center': (20, 15),
                    'size': 12,
                    'strength': 0.75,
                    'direction': 'clockwise',
                    'sign': -1
                }
            ],
            'flow_regimes': {
                'laminar': {
                    'regions': [
                        {
                            'center': (7, 7),
                            'size': 100,
                            'bounds': ((5, 5), (15, 15))
                        }
                    ],
                    'coverage': 0.25,
                    'average_flow': 0.8
                },
                'turbulent': {
                    'regions': [
                        {
                            'center': (22, 12),
                            'size': 50,
                            'bounds': ((18, 8), (25, 16))
                        }
                    ],
                    'coverage': 0.15,
                    'average_flow': 0.6,
                    'average_vorticity': 0.3
                }
            },
            'pressure_gradients': [
                {
                    'start': (15, 8),
                    'end': (8, 15),
                    'magnitude': 0.9,
                    'direction': 'decreasing'
                }
            ]
        }
        
        # 5. Create sample price and signal data for time series plotting
        self.dates = [datetime(2023, 1, 1) + timedelta(minutes=i) for i in range(100)]
        self.prices = 100 + np.cumsum(np.random.normal(0, 0.1, 100))
        self.signals = np.random.choice([-1, 0, 1], size=100)
        
        # Create dataframe
        self.market_data = pd.DataFrame({
            'DateTime': self.dates,
            'Price': self.prices,
            'Signal': self.signals
        })
        self.market_data.set_index('DateTime', inplace=True)

    def test_plot_flow_field(self):
        """Test flow field visualization."""
        # Test with default parameters
        fig, ax = plt.subplots()
        ax = plot_flow_field(self.flow_field, ax=ax)
        
        # Check that the plot was created
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_title(), "Flow Field")
        self.assertEqual(ax.get_xlabel(), "Price Level")
        self.assertEqual(ax.get_ylabel(), "Time")
        
        # Test with custom parameters
        fig, ax = plt.subplots()
        ax = plot_flow_field(
            self.flow_field, 
            ax=ax, 
            title="Custom Flow Plot", 
            cmap="plasma", 
            show_streamlines=False
        )
        
        # Check that custom parameters were applied
        self.assertEqual(ax.get_title(), "Custom Flow Plot")
        
        # Save plot to buffer to ensure it works
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # If the buffer has content, the plot was successfully created
        self.assertGreater(len(buf.getvalue()), 0)
        
        # Clean up
        plt.close(fig)

    def test_plot_vorticity_field(self):
        """Test vorticity field visualization."""
        # Test with default parameters
        fig, ax = plt.subplots()
        ax = plot_vorticity_field(self.vorticity_field, ax=ax)
        
        # Check that the plot was created
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_title(), "Vorticity Field")
        self.assertEqual(ax.get_xlabel(), "Price Level")
        self.assertEqual(ax.get_ylabel(), "Time")
        
        # Test with custom parameters
        fig, ax = plt.subplots()
        ax = plot_vorticity_field(
            self.vorticity_field, 
            ax=ax, 
            title="Custom Vorticity Plot", 
            cmap="coolwarm", 
            show_contours=False
        )
        
        # Check that custom parameters were applied
        self.assertEqual(ax.get_title(), "Custom Vorticity Plot")
        
        # Save plot to buffer to ensure it works
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # If the buffer has content, the plot was successfully created
        self.assertGreater(len(buf.getvalue()), 0)
        
        # Clean up
        plt.close(fig)

    def test_plot_pressure_field(self):
        """Test pressure field visualization."""
        # Test with default parameters
        fig, ax = plt.subplots()
        ax = plot_pressure_field(self.pressure_field, ax=ax)
        
        # Check that the plot was created
        self.assertIsNotNone(ax)
        self.assertEqual(ax.get_title(), "Pressure Field")
        self.assertEqual(ax.get_xlabel(), "Price Level")
        self.assertEqual(ax.get_ylabel(), "Time")
        
        # Test with custom parameters
        fig, ax = plt.subplots()
        ax = plot_pressure_field(
            self.pressure_field, 
            ax=ax, 
            title="Custom Pressure Plot", 
            cmap="viridis", 
            show_contours=True,
            show_gradients=True
        )
        
        # Check that custom parameters were applied
        self.assertEqual(ax.get_title(), "Custom Pressure Plot")
        
        # Save plot to buffer to ensure it works
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        
        # If the buffer has content, the plot was successfully created
        self.assertGreater(len(buf.getvalue()), 0)
        
        # Clean up
        plt.close(fig)

    def test_plot_pattern_overlay(self):
        """Test pattern overlay visualization."""
        # Test with default parameters
        fig, ax = plt.subplots()
        try:
            ax = plot_pattern_overlay(self.flow_field, self.patterns, ax=ax)
            
            # Check that the plot was created
            self.assertIsNotNone(ax)
            self.assertEqual(ax.get_title(), "Identified Patterns")
            
            # Save plot to buffer to ensure it works
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # If the buffer has content, the plot was successfully created
            self.assertGreater(len(buf.getvalue()), 0)
            
            plot_successful = True
        except Exception as e:
            plot_successful = False
        
        # Clean up
        plt.close(fig)
        
        # The test should pass if the plotting function works
        self.assertTrue(plot_successful)

    def test_plot_multiple_together(self):
        """Test plotting multiple visualizations together."""
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        
        try:
            # Plot flow field
            plot_flow_field(self.flow_field, ax=axs[0, 0], title="Flow Field")
            
            # Plot vorticity field
            plot_vorticity_field(self.vorticity_field, ax=axs[0, 1], title="Vorticity Field")
            
            # Plot pressure field
            plot_pressure_field(self.pressure_field, ax=axs[1, 0], title="Pressure Field")
            
            # Plot pattern overlay
            plot_pattern_overlay(self.flow_field, self.patterns, ax=axs[1, 1], title="Patterns")
            
            # Add figure title
            fig.suptitle("Fluid Dynamics Market Visualization", fontsize=16)
            fig.tight_layout()
            
            # Save plot to buffer to ensure it works
            buf = io.BytesIO()
            fig.savefig(buf, format='png')
            buf.seek(0)
            
            # If the buffer has content, the plot was successfully created
            self.assertGreater(len(buf.getvalue()), 0)
            
            plot_successful = True
        except Exception as e:
            plot_successful = False
        
        # Clean up
        plt.close(fig)
        
        # The test should pass if the plotting function works
        self.assertTrue(plot_successful)


if __name__ == '__main__':
    unittest.main() 