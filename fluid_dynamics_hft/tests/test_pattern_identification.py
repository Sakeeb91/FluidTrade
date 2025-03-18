import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fluid_dynamics_hft.src.pattern_identification import FluidPatternRecognizer

class TestFluidPatternRecognizer(unittest.TestCase):
    """Unit tests for the FluidPatternRecognizer class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Initialize pattern recognizer
        self.recognizer = FluidPatternRecognizer()
        
        # Create synthetic test fields for pattern detection
        np.random.seed(42)
        self.time_steps, self.price_levels = 30, 20
        
        # 1. Create flow field with some structured patterns
        self.flow_field = np.zeros((self.time_steps, self.price_levels))
        
        # Add a laminar flow region (parallel flow)
        for i in range(5, 15):
            for j in range(5, 15):
                self.flow_field[i, j] = 0.8
        
        # Add some random noise
        self.flow_field += np.random.normal(0, 0.1, (self.time_steps, self.price_levels))
        
        # 2. Create a vorticity field with vortex patterns
        self.vorticity_field = np.zeros((self.time_steps, self.price_levels))
        
        # Add a positive vortex (counterclockwise)
        center_i, center_j = 10, 10
        radius = 3
        for i in range(self.time_steps):
            for j in range(self.price_levels):
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if dist < radius:
                    self.vorticity_field[i, j] = 0.9 * (1 - dist/radius)
        
        # Add a negative vortex (clockwise)
        center_i, center_j = 20, 15
        radius = 2
        for i in range(self.time_steps):
            for j in range(self.price_levels):
                dist = np.sqrt((i - center_i)**2 + (j - center_j)**2)
                if dist < radius:
                    self.vorticity_field[i, j] = -0.8 * (1 - dist/radius)
        
        # Add some random noise
        self.vorticity_field += np.random.normal(0, 0.05, (self.time_steps, self.price_levels))
        
        # 3. Create a pressure field with gradients
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

    def test_initialization(self):
        """Test that the pattern recognizer initializes correctly."""
        # Check that patterns dictionary is initialized
        self.assertEqual(self.recognizer.patterns, {})
        
        # Check that pattern definitions exist
        self.assertIsNotNone(self.recognizer.pattern_definitions)
        self.assertIsInstance(self.recognizer.pattern_definitions, dict)
        
        # Check for expected pattern types
        expected_patterns = ['vortex', 'laminar_flow', 'turbulent_flow', 'pressure_gradient', 
                            'shock_wave', 'boundary_layer', 'eddy_current', 'jet_stream']
        for pattern in expected_patterns:
            self.assertIn(pattern, self.recognizer.pattern_definitions)

    def test_identify_vortices(self):
        """Test vortex identification in the vorticity field."""
        # Call the method to identify vortices
        vortices = self.recognizer.identify_vortices(self.vorticity_field, threshold=0.6)
        
        # Check that vortices were detected
        self.assertIsInstance(vortices, list)
        self.assertGreater(len(vortices), 0)
        
        # Verify vortex properties
        for vortex in vortices:
            self.assertIn('center', vortex)
            self.assertIn('size', vortex)
            self.assertIn('strength', vortex)
            self.assertIn('direction', vortex)
            self.assertIn('sign', vortex)
            
            # Check direction string matches sign
            if vortex['sign'] > 0:
                self.assertEqual(vortex['direction'], 'counterclockwise')
            else:
                self.assertEqual(vortex['direction'], 'clockwise')

    def test_identify_flow_regimes(self):
        """Test identification of flow regimes (laminar vs turbulent)."""
        # Call the method to identify flow regimes
        regimes = self.recognizer.identify_flow_regimes(self.flow_field, self.vorticity_field)
        
        # Check that regimes were identified
        self.assertIsInstance(regimes, dict)
        self.assertIn('laminar', regimes)
        self.assertIn('turbulent', regimes)
        self.assertIn('transitional', regimes)
        
        # Check that regime properties are present
        self.assertIn('mask', regimes['laminar'])
        self.assertIn('coverage', regimes['laminar'])
        self.assertIn('average_flow', regimes['laminar'])
        
        self.assertIn('mask', regimes['turbulent'])
        self.assertIn('coverage', regimes['turbulent'])
        self.assertIn('average_flow', regimes['turbulent'])
        self.assertIn('average_vorticity', regimes['turbulent'])
        
        # Verify shape of regime masks
        self.assertEqual(regimes['laminar']['mask'].shape, self.flow_field.shape)
        self.assertEqual(regimes['turbulent']['mask'].shape, self.flow_field.shape)
        
        # Check flow regime definitions make sense (laminar should have low vorticity)
        if regimes['laminar']['regions'] and regimes['turbulent']['regions']:
            laminar_region = regimes['laminar']['regions'][0]
            turbulent_region = regimes['turbulent']['regions'][0]
            
            y1, x1 = laminar_region['center']
            y2, x2 = turbulent_region['center']
            
            avg_vorticity_laminar = np.mean(np.abs(self.vorticity_field[int(y1), int(x1)]))
            avg_vorticity_turbulent = np.mean(np.abs(self.vorticity_field[int(y2), int(x2)]))
            
            # Turbulent regions should have higher vorticity than laminar
            if avg_vorticity_laminar > 0 and avg_vorticity_turbulent > 0:
                self.assertLess(avg_vorticity_laminar, avg_vorticity_turbulent)

    def test_identify_pressure_gradients(self):
        """Test identification of pressure gradients."""
        # Call the method to identify pressure gradients
        gradients = self.recognizer.identify_pressure_gradients(self.pressure_field, threshold=0.5)
        
        # Check that gradients were identified
        self.assertIsInstance(gradients, list)
        
        # If gradients were found, verify their properties
        if gradients:
            gradient = gradients[0]
            self.assertIn('start', gradient)
            self.assertIn('end', gradient)
            self.assertIn('magnitude', gradient)
            self.assertIn('direction', gradient)
            
            # Verify gradient magnitude is positive
            self.assertGreater(gradient['magnitude'], 0)

    def test_identify_patterns(self):
        """Test the main pattern identification functionality."""
        # Call the method to identify all patterns
        patterns = self.recognizer.identify_patterns(
            self.flow_field, 
            self.vorticity_field, 
            self.pressure_field
        )
        
        # Check that patterns were identified
        self.assertIsInstance(patterns, dict)
        
        # Check for expected pattern categories
        expected_categories = ['vortices', 'flow_regimes', 'pressure_gradients', 'shock_waves']
        for category in expected_categories:
            self.assertIn(category, patterns)
            
        # Check that flow regimes are identified
        self.assertIn('laminar', patterns['flow_regimes'])
        self.assertIn('turbulent', patterns['flow_regimes'])

    def test_generate_market_insights(self):
        """Test the generation of market insights from patterns."""
        # First identify patterns
        patterns = self.recognizer.identify_patterns(
            self.flow_field, 
            self.vorticity_field, 
            self.pressure_field
        )
        
        # Generate insights
        insights = self.recognizer.generate_market_insights(patterns)
        
        # Check that insights were generated
        self.assertIsInstance(insights, dict)
        
        # Check for expected insight categories
        expected_categories = ['vortices', 'flow_regimes', 'pressure_gradients', 'market_state']
        for category in expected_categories:
            self.assertIn(category, insights)
            
        # Check market state assessment
        self.assertIn('regime', insights['market_state'])
        self.assertIn('volatility', insights['market_state'])
        self.assertIn('trend', insights['market_state'])
        self.assertIn('reversal_probability', insights['market_state'])

    def test_visualize_patterns(self):
        """Test the pattern visualization functionality."""
        # First identify patterns
        patterns = self.recognizer.identify_patterns(
            self.flow_field, 
            self.vorticity_field, 
            self.pressure_field
        )
        
        # Create a test figure for visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Test visualization method
        try:
            self.recognizer.visualize_patterns(
                self.flow_field,
                self.vorticity_field,
                self.pressure_field,
                patterns,
            )
            plt.close(fig)  # Close figure to avoid display
            visualization_success = True
        except Exception as e:
            visualization_success = False
            plt.close(fig)  # Close figure to avoid display
        
        self.assertTrue(visualization_success)


if __name__ == '__main__':
    unittest.main()
