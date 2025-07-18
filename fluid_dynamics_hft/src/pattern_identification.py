import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import gaussian_filter
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import networkx as nx
from scipy.stats import entropy
from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib.collections import PatchCollection
import warnings
from skimage import morphology
warnings.filterwarnings('ignore')

class FluidPatternRecognizer:
    """
    Class for identifying and classifying market microstructure patterns
    using fluid dynamics concepts.
    """
    
    def __init__(self, config=None):
        """Initialize the pattern recognizer"""
        self.config = config or {}
        self.patterns = {}
        self.pattern_definitions = self._define_patterns()
        self.initialized = False
    
    def initialize(self):
        """Initialize the pattern recognizer."""
        self.initialized = True
    
    def detect_patterns(self, flow_field, pressure_field, vorticity_field):
        """
        Detect all fluid dynamics patterns in the given fields.
        
        Args:
            flow_field: 2D numpy array of flow values
            pressure_field: 2D numpy array of pressure values  
            vorticity_field: 2D numpy array of vorticity values
            
        Returns:
            Dictionary of detected patterns
        """
        patterns = {}
        
        # Detect vortices
        vortices = self.identify_vortices(vorticity_field)
        if vortices:
            patterns['vortices'] = vortices
        
        # Detect flow regimes
        regimes = self.identify_flow_regimes(flow_field, vorticity_field)
        patterns.update(regimes)
        
        # Detect pressure patterns
        pressure_patterns = self.identify_pressure_patterns(pressure_field)
        patterns.update(pressure_patterns)
        
        return patterns
    
    def identify_pressure_patterns(self, pressure_field):
        """
        Identify pressure-based patterns.
        
        Args:
            pressure_field: 2D numpy array of pressure values
            
        Returns:
            Dictionary of pressure patterns
        """
        patterns = {}
        
        # Calculate pressure gradient
        grad_y, grad_x = np.gradient(pressure_field)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find high gradient regions
        high_gradient_threshold = np.percentile(gradient_magnitude, 90)
        high_gradient_regions = gradient_magnitude > high_gradient_threshold
        
        if np.any(high_gradient_regions):
            patterns['pressure_gradients'] = {
                'regions': high_gradient_regions,
                'magnitude': gradient_magnitude,
                'max_gradient': np.max(gradient_magnitude)
            }
        
        return patterns
    
    def _define_patterns(self):
        """Define common fluid dynamics patterns and their market interpretations"""
        patterns = {
            # Vortex patterns
            'vortex': {
                'description': 'Circular flow pattern indicating potential trend reversal',
                'fluid_signature': 'High absolute vorticity with circular flow vectors',
                'market_interpretation': 'Price consolidation followed by trend reversal',
                'typical_duration': 'Short to medium term (minutes to hours)',
                'trading_implication': 'Enter counter-trend positions at vortex formation'
            },
            
            # Laminar flow
            'laminar_flow': {
                'description': 'Smooth, parallel flow pattern indicating stable trend',
                'fluid_signature': 'Low vorticity, high flow magnitude, parallel vectors',
                'market_interpretation': 'Strong trending market with minimal volatility',
                'typical_duration': 'Medium term (hours to days)',
                'trading_implication': 'Enter with-trend positions during laminar flow'
            },
            
            # Turbulent flow
            'turbulent_flow': {
                'description': 'Chaotic flow pattern indicating high volatility',
                'fluid_signature': 'High vorticity, irregular flow patterns',
                'market_interpretation': 'Choppy market with high volatility',
                'typical_duration': 'Short term (seconds to minutes)',
                'trading_implication': 'Reduce position sizes, widen stops, or stay out'
            },
            
            # Pressure gradient
            'pressure_gradient': {
                'description': 'Steep change in pressure indicating potential momentum shift',
                'fluid_signature': 'High gradient magnitude in pressure field',
                'market_interpretation': 'Buildup of buying/selling pressure before price move',
                'typical_duration': 'Short term (seconds to minutes)',
                'trading_implication': 'Enter positions in direction of pressure gradient'
            },
            
            # Shock wave
            'shock_wave': {
                'description': 'Sudden, sharp discontinuity in flow field',
                'fluid_signature': 'Abrupt change in flow direction/magnitude',
                'market_interpretation': 'Sudden price shock due to news or large orders',
                'typical_duration': 'Very short term (milliseconds to seconds)',
                'trading_implication': 'Potential for rapid mean reversion after shock'
            },
            
            # Boundary layer
            'boundary_layer': {
                'description': 'Thin layer where flow velocity changes rapidly',
                'fluid_signature': 'High shear stress near boundaries',
                'market_interpretation': 'Support/resistance levels with order clustering',
                'typical_duration': 'Medium to long term (hours to days)',
                'trading_implication': 'Place orders at boundary layer edges'
            },
            
            # Eddy current
            'eddy_current': {
                'description': 'Small circular current against main flow',
                'fluid_signature': 'Small vortices within larger flow pattern',
                'market_interpretation': 'Short counter-trend move within larger trend',
                'typical_duration': 'Short term (minutes to hours)',
                'trading_implication': 'Potential counter-trend scalping opportunity'
            },
            
            # Jet stream
            'jet_stream': {
                'description': 'Narrow, high-velocity flow between slower regions',
                'fluid_signature': 'Localized high flow magnitude in narrow channel',
                'market_interpretation': 'Rapid directional movement with high momentum',
                'typical_duration': 'Short to medium term (minutes to hours)',
                'trading_implication': 'Enter with-trend positions at jet stream formation'
            }
        }
        
        return patterns
    
    def identify_vortices(self, vorticity_field, threshold=0.7):
        """
        Identify vortex patterns in the vorticity field.
        
        Parameters:
        -----------
        vorticity_field : numpy.ndarray
            2D array of vorticity values
        threshold : float
            Threshold for vortex detection (0.0 to 1.0)
            
        Returns:
        --------
        vortices : list of dict
            List of detected vortices with properties
        """
        # Find regions of high absolute vorticity
        high_vorticity = np.abs(vorticity_field) > threshold
        
        if not np.any(high_vorticity):
            return []
        
        # Label connected regions
        from scipy import ndimage
        labeled_regions, num_regions = ndimage.label(high_vorticity)
        
        vortices = []
        
        # Extract properties of each vortex
        for i in range(1, num_regions + 1):
            # Get region mask
            region = labeled_regions == i
            
            # Get region properties
            y_indices, x_indices = np.where(region)
            center_y = np.mean(y_indices)
            center_x = np.mean(x_indices)
            size = np.sum(region)
            
            # Get vorticity sign (positive = counterclockwise, negative = clockwise)
            vorticity_values = vorticity_field[region]
            vorticity_sign = np.sign(np.mean(vorticity_values))
            vorticity_strength = np.mean(np.abs(vorticity_values))
            
            # Create vortex object
            vortex = {
                'center': (center_y, center_x),
                'size': size,
                'strength': vorticity_strength,
                'direction': 'counterclockwise' if vorticity_sign > 0 else 'clockwise',
                'sign': vorticity_sign
            }
            
            vortices.append(vortex)
        
        return vortices
    
    def identify_flow_regimes(self, flow_field, vorticity_field, thresholds=None):
        """
        Identify laminar vs turbulent flow regimes.
        
        Parameters:
        -----------
        flow_field : numpy.ndarray
            2D array of flow magnitudes
        vorticity_field : numpy.ndarray
            2D array of vorticity values
        thresholds : dict, optional
            Dictionary of thresholds for detection
            
        Returns:
        --------
        regimes : dict
            Dictionary with flow regime masks and properties
        """
        if thresholds is None:
            thresholds = {
                'flow': 0.5,          # Threshold for significant flow
                'vorticity': 0.4,     # Threshold for significant vorticity
                'laminar': 0.3,       # Max vorticity for laminar flow
                'turbulent': 0.6      # Min vorticity for turbulent flow
            }
        
        # Calculate flow magnitude
        flow_magnitude = np.abs(flow_field)
        
        # Identify regions with significant flow
        significant_flow = flow_magnitude > thresholds['flow']
        
        # Identify flow regimes
        laminar_mask = (np.abs(vorticity_field) < thresholds['laminar']) & significant_flow
        turbulent_mask = (np.abs(vorticity_field) > thresholds['turbulent']) & significant_flow
        
        # Calculate regime properties
        regimes = {
            'laminar': {
                'mask': laminar_mask,
                'coverage': np.sum(laminar_mask) / flow_field.size,
                'average_flow': np.mean(flow_magnitude[laminar_mask]) if np.any(laminar_mask) else 0,
                'regions': None
            },
            'turbulent': {
                'mask': turbulent_mask,
                'coverage': np.sum(turbulent_mask) / flow_field.size,
                'average_flow': np.mean(flow_magnitude[turbulent_mask]) if np.any(turbulent_mask) else 0,
                'average_vorticity': np.mean(np.abs(vorticity_field[turbulent_mask])) if np.any(turbulent_mask) else 0,
                'regions': None
            },
            'transitional': {
                'mask': significant_flow & ~(laminar_mask | turbulent_mask),
                'coverage': np.sum(significant_flow & ~(laminar_mask | turbulent_mask)) / flow_field.size
            }
        }
        
        # Label connected regions
        from scipy import ndimage
        for regime in ['laminar', 'turbulent']:
            labeled_regions, num_regions = ndimage.label(regimes[regime]['mask'])
            
            regions = []
            for i in range(1, num_regions + 1):
                region_mask = labeled_regions == i
                y_indices, x_indices = np.where(region_mask)
                
                region = {
                    'center': (np.mean(y_indices), np.mean(x_indices)),
                    'size': np.sum(region_mask),
                    'bounds': (
                        (min(y_indices), min(x_indices)),
                        (max(y_indices), max(x_indices))
                    )
                }
                
                regions.append(region)
            
            regimes[regime]['regions'] = regions
        
        return regimes
    
    def identify_pressure_gradients(self, pressure_field, threshold=0.5):
        """
        Identify significant pressure gradients.
        
        Parameters:
        -----------
        pressure_field : numpy.ndarray
            2D array of pressure values
        threshold : float
            Threshold for gradient detection (0.0 to 1.0)
            
        Returns:
        --------
        gradients : dict
            Dictionary with gradient information
        """
        # Calculate pressure gradients
        grad_y, grad_x = np.gradient(pressure_field)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_y**2 + grad_x**2)
        
        # Normalize gradient magnitude
        if np.max(gradient_magnitude) > 0:
            gradient_magnitude = gradient_magnitude / np.max(gradient_magnitude)
        
        # Identify significant gradients
        significant_gradients = gradient_magnitude > threshold
        
        # Calculate gradient direction
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # Label connected gradient regions
        from scipy import ndimage
        labeled_regions, num_regions = ndimage.label(significant_gradients)
        
        gradient_regions = []
        
        for i in range(1, num_regions + 1):
            region_mask = labeled_regions == i
            y_indices, x_indices = np.where(region_mask)
            
            region = {
                'center': (np.mean(y_indices), np.mean(x_indices)),
                'size': np.sum(region_mask),
                'magnitude': np.mean(gradient_magnitude[region_mask]),
                'direction': np.mean(gradient_direction[region_mask]),
                'bounds': (
                    (min(y_indices), min(x_indices)),
                    (max(y_indices), max(x_indices))
                )
            }
            
            gradient_regions.append(region)
        
        gradients = {
            'field': (grad_y, grad_x),
            'magnitude': gradient_magnitude,
            'direction': gradient_direction,
            'significant_mask': significant_gradients,
            'regions': gradient_regions,
            'coverage': np.sum(significant_gradients) / pressure_field.size,
            'num_regions': num_regions
        }
        
        return gradients
    
    def identify_patterns(self, flow_field, vorticity_field, pressure_field):
        """
        Identify all major fluid dynamics patterns in the given fields.
        
        Parameters:
        -----------
        flow_field : numpy.ndarray
            2D array of flow magnitudes
        vorticity_field : numpy.ndarray
            2D array of vorticity values
        pressure_field : numpy.ndarray
            2D array of pressure values
            
        Returns:
        --------
        patterns : dict
            Dictionary containing all identified patterns
        """
        patterns = {}
        
        # Identify vortices
        vortices = self.identify_vortices(vorticity_field)
        if vortices:
            patterns['vortices'] = vortices
        
        # Identify flow regimes
        regimes = self.identify_flow_regimes(flow_field, vorticity_field)
        patterns.update(regimes)
        
        # Identify pressure gradients
        gradients = self.identify_pressure_gradients(pressure_field)
        patterns['pressure_gradients'] = gradients
        
        return patterns
    
    def generate_market_insights(self, patterns):
        """
        Generate market insights based on identified patterns.
        
        Parameters:
        -----------
        patterns : dict
            Dictionary of identified patterns
            
        Returns:
        --------
        insights : dict
            Dictionary of market insights and trading implications
        """
        insights = {
            'market_state': {},
            'vortex_analysis': {'insights': []},
            'flow_analysis': {'insights': []},
            'pressure_analysis': {'insights': []}
        }
        
        # Analyze vortices
        if 'vortices' in patterns:
            vortices = patterns['vortices']
            insights['vortex_analysis']['count'] = len(vortices)
            
            if len(vortices) > 0:
                insights['vortex_analysis']['insights'].append(
                    f"Detected {len(vortices)} vortex pattern(s) - potential reversal zones"
                )
                
                # Analyze vortex strength
                avg_strength = np.mean([v['strength'] for v in vortices])
                if avg_strength > 0.7:
                    insights['vortex_analysis']['insights'].append(
                        "High vortex strength detected - strong reversal potential"
                    )
                elif avg_strength > 0.4:
                    insights['vortex_analysis']['insights'].append(
                        "Moderate vortex strength - watch for trend changes"
                    )
        
        # Analyze flow regimes
        if 'laminar' in patterns:
            laminar = patterns['laminar']
            if laminar['coverage'] > 0.3:
                insights['flow_analysis']['insights'].append(
                    f"Laminar flow covers {laminar['coverage']:.1%} of market - trending conditions"
                )
        
        if 'turbulent' in patterns:
            turbulent = patterns['turbulent']
            if turbulent['coverage'] > 0.2:
                insights['flow_analysis']['insights'].append(
                    f"Turbulent flow covers {turbulent['coverage']:.1%} of market - volatile conditions"
                )
        
        # Analyze pressure gradients
        if 'pressure_gradients' in patterns:
            gradients = patterns['pressure_gradients']
            if gradients['num_regions'] > 0:
                insights['pressure_analysis']['insights'].append(
                    f"Detected {gradients['num_regions']} pressure gradient region(s) - momentum building"
                )
        
        # Generate market state summary
        insights['market_state'] = self._generate_market_state_summary(patterns)
        
        return insights
    
    def _generate_market_state_summary(self, patterns):
        """Generate overall market state summary."""
        state = {
            'regime': 'neutral',
            'volatility': 'moderate',
            'momentum': 'sideways',
            'key_points': []
        }
        
        # Determine regime
        if 'laminar' in patterns and patterns['laminar']['coverage'] > 0.4:
            state['regime'] = 'trending'
            state['momentum'] = 'directional'
        elif 'turbulent' in patterns and patterns['turbulent']['coverage'] > 0.3:
            state['regime'] = 'volatile'
            state['volatility'] = 'high'
        
        # Check for vortices
        if 'vortices' in patterns and len(patterns['vortices']) > 0:
            state['key_points'].append('Vortex patterns detected - potential reversal zones')
        
        # Check for pressure gradients
        if 'pressure_gradients' in patterns and patterns['pressure_gradients']['num_regions'] > 0:
            state['key_points'].append('Pressure gradients detected - momentum building')
        
        return state
    
    def visualize_patterns(self, flow_field, vorticity_field, pressure_field, patterns):
        """
        Visualize identified patterns.
        
        Parameters:
        -----------
        flow_field : numpy.ndarray
            2D array of flow magnitudes
        vorticity_field : numpy.ndarray
            2D array of vorticity values
        pressure_field : numpy.ndarray
            2D array of pressure values
        patterns : dict
            Dictionary of identified patterns
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot flow field
        im1 = axes[0, 0].imshow(flow_field, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Flow Field')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Plot vorticity field with vortices
        im2 = axes[0, 1].imshow(vorticity_field, cmap='RdBu', aspect='auto')
        axes[0, 1].set_title('Vorticity Field')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Highlight vortices
        if 'vortices' in patterns:
            for vortex in patterns['vortices']:
                center = vortex['center']
                circle = plt.Circle((center[1], center[0]), radius=2, 
                                  fill=False, color='yellow', linewidth=2)
                axes[0, 1].add_patch(circle)
        
        # Plot pressure field
        im3 = axes[1, 0].imshow(pressure_field, cmap='plasma', aspect='auto')
        axes[1, 0].set_title('Pressure Field')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Plot regime classification
        regime_map = np.zeros_like(flow_field)
        if 'laminar' in patterns:
            regime_map[patterns['laminar']['mask']] = 1
        if 'turbulent' in patterns:
            regime_map[patterns['turbulent']['mask']] = 2
        
        im4 = axes[1, 1].imshow(regime_map, cmap='Set1', aspect='auto')
        axes[1, 1].set_title('Flow Regimes (0=neutral, 1=laminar, 2=turbulent)')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.show()


def test_pattern_recognizer():
    """Test the pattern recognizer with synthetic data."""
    print("Testing FluidPatternRecognizer...")
    
    # Generate synthetic test data
    x = np.linspace(0, 1, 30)
    y = np.linspace(0, 1, 20)
    X, Y = np.meshgrid(x, y)
    
    # Create vortex pattern
    r = np.sqrt((X - 0.3)**2 + (Y - 0.3)**2)
    theta = np.arctan2(Y - 0.3, X - 0.3)
    
    # Vortex component
    vortex_x = -np.sin(theta) * np.exp(-10*r)
    vortex_y = np.cos(theta) * np.exp(-10*r)
    
    # Laminar component
    laminar_x = 0.5 * (X > 0.5)
    laminar_y = 0.0
    
    # Combine components
    flow_x = vortex_x + laminar_x
    flow_y = vortex_y + laminar_y
    
    # Calculate flow field (magnitude)
    flow_field = np.sqrt(flow_x**2 + flow_y**2)
    
    # Calculate vorticity field
    dx_y, dy_y = np.gradient(flow_y)
    dx_x, dy_x = np.gradient(flow_x)
    vorticity_field = dx_y - dy_x
    
    # Create pressure field
    pressure_field = np.exp(-5*((X-0.7)**2 + (Y-0.7)**2)) - np.exp(-10*((X-0.2)**2 + (Y-0.5)**2))
    
    # Initialize pattern recognizer
    recognizer = FluidPatternRecognizer()
    
    # Identify patterns
    patterns = recognizer.identify_patterns(flow_field, vorticity_field, pressure_field)
    
    # Generate insights
    insights = recognizer.generate_market_insights(patterns)
    
    # Print insights
    print("\n=== MARKET INSIGHTS ===")
    for category, category_insights in insights.items():
        print(f"\n{category.upper()}:")
        if category == 'market_state':
            for key, value in category_insights.items():
                if key != 'key_points':
                    print(f"  {key}: {value}")
            print(f"  key_points: {len(category_insights.get('key_points', []))} points identified")
        elif isinstance(category_insights, dict) and 'insights' in category_insights:
            for insight in category_insights['insights']:
                print(f"  - {insight}")
    
    # Visualize patterns
    recognizer.visualize_patterns(flow_field, vorticity_field, pressure_field, patterns)


if __name__ == "__main__":
    test_pattern_recognizer()
