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
    
    def __init__(self):
        """Initialize the pattern recognizer"""
        self.patterns = {}
        self.pattern_definitions = self._define_patterns()
    
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
            'coverage': np.sum(significant_gradients) / pressure_field.size
        }
        
        return gradients
    
    def identify_shock_waves(self, flow_field, threshold=0.7, window_size=3):
        """
        Identify shock waves (sudden discontinuities in flow).
        
        Parameters:
        -----------
        flow_field : numpy.ndarray
            2D array of flow values
        threshold : float
            Threshold for shock detection (0.0 to 1.0)
        window_size : int
            Size of window to use for detecting discontinuities
            
        Returns:
        --------
        shocks : dict
            Dictionary with shock wave information
        """
        # Calculate flow derivatives
        dy, dx = np.gradient(flow_field)
        
        # Calculate total derivative magnitude
        derivative_magnitude = np.sqrt(dy**2 + dx**2)
        
        # Normalize
        if np.max(derivative_magnitude) > 0:
            derivative_magnitude = derivative_magnitude / np.max(derivative_magnitude)
        
        # Use a more focused detection approach for sharp discontinuities
        shock_detector = np.zeros_like(flow_field)
        
        # Iterate through interior points
        for i in range(window_size, flow_field.shape[0] - window_size):
            for j in range(window_size, flow_field.shape[1] - window_size):
                # Get values in window
                window = flow_field[i-window_size:i+window_size+1, j-window_size:j+window_size+1]
                
                # Calculate max difference in window
                max_diff = np.max(window) - np.min(window)
                
                # Calculate rate of change
                rate_of_change = max_diff / (2 * window_size)
                
                # Normalize by local average
                local_avg = np.mean(np.abs(window))
                if local_avg > 0:
                    normalized_change = rate_of_change / local_avg
                    shock_detector[i, j] = normalized_change
        
        # Normalize shock detector
        if np.max(shock_detector) > 0:
            shock_detector = shock_detector / np.max(shock_detector)
        
        # Threshold to find shocks
        shock_mask = shock_detector > threshold
        
        # Label connected shock regions
        from scipy import ndimage
        labeled_regions, num_regions = ndimage.label(shock_mask)
        
        shock_regions = []
        
        for i in range(1, num_regions + 1):
            region_mask = labeled_regions == i
            y_indices, x_indices = np.where(region_mask)
            
            if len(y_indices) > 0:
                region = {
                    'center': (np.mean(y_indices), np.mean(x_indices)),
                    'size': np.sum(region_mask),
                    'strength': np.mean(shock_detector[region_mask]),
                    'bounds': (
                        (min(y_indices), min(x_indices)),
                        (max(y_indices), max(x_indices))
                    )
                }
                
                shock_regions.append(region)
        
        shocks = {
            'detector': shock_detector,
            'mask': shock_mask,
            'regions': shock_regions,
            'count': len(shock_regions),
            'coverage': np.sum(shock_mask) / flow_field.size
        }
        
        return shocks
    
    def identify_boundary_layers(self, flow_field, vorticity_field, threshold=0.6):
        """
        Identify boundary layers (regions of high shear).
        
        Parameters:
        -----------
        flow_field : numpy.ndarray
            2D array of flow values
        vorticity_field : numpy.ndarray
            2D array of vorticity values
        threshold : float
            Threshold for boundary layer detection (0.0 to 1.0)
            
        Returns:
        --------
        boundary_layers : dict
            Dictionary with boundary layer information
        """
        # Boundary layers occur where flow changes rapidly over small distances
        # They can be identified by high vorticity and flow gradients
        
        # Calculate flow gradients
        flow_dy, flow_dx = np.gradient(flow_field)
        
        # Calculate gradient magnitude
        flow_gradient_magnitude = np.sqrt(flow_dy**2 + flow_dx**2)
        
        # Normalize
        if np.max(flow_gradient_magnitude) > 0:
            flow_gradient_magnitude = flow_gradient_magnitude / np.max(flow_gradient_magnitude)
        
        # Boundary layer detection combines flow gradient and vorticity
        # This is a simplified approach - real fluid boundary layers are more complex
        abs_vorticity = np.abs(vorticity_field)
        if np.max(abs_vorticity) > 0:
            abs_vorticity = abs_vorticity / np.max(abs_vorticity)
        
        # Combined detector
        boundary_detector = (flow_gradient_magnitude + abs_vorticity) / 2
        
        # Threshold to find boundary layers
        boundary_mask = boundary_detector > threshold
        
        # Thin the boundaries
        thinned_mask = morphology.skeletonize(boundary_mask)
        
        # Find connected components
        from scipy import ndimage
        labeled_regions, num_regions = ndimage.label(thinned_mask)
        
        boundary_regions = []
        
        for i in range(1, num_regions + 1):
            region_mask = labeled_regions == i
            y_indices, x_indices = np.where(region_mask)
            
            if len(y_indices) > 0:
                # Calculate boundary orientation
                if len(y_indices) > 1:
                    # Use principal component analysis for orientation
                    coords = np.vstack((y_indices, x_indices)).T
                    coords_centered = coords - np.mean(coords, axis=0)
                    cov = np.cov(coords_centered.T)
                    evals, evecs = np.linalg.eig(cov)
                    
                    # Principal direction
                    main_axis = evecs[:, np.argmax(evals)]
                    angle = np.arctan2(main_axis[1], main_axis[0])
                    
                    # Length along main axis
                    projection = np.dot(coords_centered, main_axis)
                    length = np.max(projection) - np.min(projection)
                else:
                    angle = 0
                    length = 1
                
                region = {
                    'points': list(zip(y_indices, x_indices)),
                    'center': (np.mean(y_indices), np.mean(x_indices)),
                    'size': len(y_indices),
                    'strength': np.mean(boundary_detector[region_mask]),
                    'orientation': angle,
                    'length': length
                }
                
                boundary_regions.append(region)
        
        boundary_layers = {
            'detector': boundary_detector,
            'mask': boundary_mask,
            'thinned_mask': thinned_mask,
            'regions': boundary_regions,
            'count': len(boundary_regions),
            'coverage': np.sum(boundary_mask) / flow_field.size
        }
        
        return boundary_layers
    
    def identify_patterns(self, flow_field, vorticity_field, pressure_field):
        """
        Identify all patterns in the provided fields.
        
        Parameters:
        -----------
        flow_field : numpy.ndarray
            2D array of flow values
        vorticity_field : numpy.ndarray
            2D array of vorticity values
        pressure_field : numpy.ndarray
            2D array of pressure values
            
        Returns:
        --------
        patterns : dict
            Dictionary with all identified patterns
        """
        patterns = {}
        
        # 1. Identify vortices
        patterns['vortices'] = self.identify_vortices(vorticity_field)
        
        # 2. Identify flow regimes (laminar vs turbulent)
        patterns['flow_regimes'] = self.identify_flow_regimes(flow_field, vorticity_field)
        
        # 3. Identify pressure gradients
        patterns['pressure_gradients'] = self.identify_pressure_gradients(pressure_field)
        
        # 4. Identify shock waves
        patterns['shock_waves'] = self.identify_shock_waves(flow_field)
        
        # 5. Identify boundary layers
        patterns['boundary_layers'] = self.identify_boundary_layers(flow_field, vorticity_field)
        
        # Save patterns
        self.patterns = patterns
        
        return patterns
    
    def generate_market_insights(self, patterns=None):
        """
        Generate market insights from identified patterns.
        
        Parameters:
        -----------
        patterns : dict, optional
            Dictionary of identified patterns. If None, uses self.patterns
            
        Returns:
        --------
        insights : dict
            Dictionary with market insights derived from fluid patterns
        """
        if patterns is None:
            patterns = self.patterns
        
        if not patterns:
            return {"error": "No patterns identified"}
        
        insights = {}
        
        # 1. Generate insights from vortices
        insights['vortex_insights'] = self._insights_from_vortices(patterns.get('vortices', []))
        
        # 2. Generate insights from flow regimes
        insights['flow_regime_insights'] = self._insights_from_flow_regimes(patterns.get('flow_regimes', {}))
        
        # 3. Generate insights from pressure gradients
        insights['pressure_insights'] = self._insights_from_pressure_gradients(patterns.get('pressure_gradients', {}))
        
        # 4. Generate insights from shock waves
        insights['shock_insights'] = self._insights_from_shocks(patterns.get('shock_waves', {}))
        
        # 5. Generate insights from boundary layers
        insights['boundary_insights'] = self._insights_from_boundaries(patterns.get('boundary_layers', {}))
        
        # 6. Generate overall market state assessment
        insights['market_state'] = self._assess_market_state(patterns, insights)
        
        return insights
    
    def _insights_from_vortices(self, vortices):
        """Generate insights from vortex patterns"""
        if not vortices:
            return {"message": "No significant vortices detected"}
        
        insights = {
            "count": len(vortices),
            "average_strength": np.mean([v['strength'] for v in vortices]),
            "clockwise_count": sum(1 for v in vortices if v['direction'] == 'clockwise'),
            "counterclockwise_count": sum(1 for v in vortices if v['direction'] == 'counterclockwise'),
            "insights": []
        }
        
        # Generate specific insights
        if insights["clockwise_count"] > insights["counterclockwise_count"]:
            insights["insights"].append("Predominance of clockwise vortices suggests potential downward price pressure")
        elif insights["counterclockwise_count"] > insights["clockwise_count"]:
            insights["insights"].append("Predominance of counterclockwise vortices suggests potential upward price pressure")
        
        if len(vortices) > 3:
            insights["insights"].append("Multiple vortices indicate a highly rotational market with potential for choppy price action")
        
        # Sort vortices by strength
        strong_vortices = [v for v in vortices if v['strength'] > 0.8]
        if strong_vortices:
            insights["insights"].append(f"Found {len(strong_vortices)} strong vortices indicating potential reversal points")
        
        return insights
    
    def _insights_from_flow_regimes(self, flow_regimes):
        """Generate insights from flow regime patterns"""
        if not flow_regimes:
            return {"message": "No flow regime information available"}
        
        laminar = flow_regimes.get('laminar', {})
        turbulent = flow_regimes.get('turbulent', {})
        transitional = flow_regimes.get('transitional', {})
        
        insights = {
            "laminar_coverage": laminar.get('coverage', 0) * 100,
            "turbulent_coverage": turbulent.get('coverage', 0) * 100,
            "transitional_coverage": transitional.get('coverage', 0) * 100,
            "dominant_regime": "none",
            "insights": []
        }
        
        # Determine dominant regime
        coverages = {
            "laminar": insights["laminar_coverage"],
            "turbulent": insights["turbulent_coverage"],
            "transitional": insights["transitional_coverage"]
        }
        insights["dominant_regime"] = max(coverages, key=coverages.get)
        
        # Generate specific insights
        if insights["dominant_regime"] == "laminar":
            insights["insights"].append("Predominantly laminar flow indicates a strong trending market with low volatility")
            if laminar.get('average_flow', 0) > 0.7:
                insights["insights"].append("High laminar flow strength suggests powerful directional momentum")
        
        elif insights["dominant_regime"] == "turbulent":
            insights["insights"].append("Predominantly turbulent flow indicates a choppy market with high volatility")
            if turbulent.get('average_vorticity', 0) > 0.7:
                insights["insights"].append("High turbulence intensity suggests extreme volatility and potential for rapid reversals")
        
        elif insights["dominant_regime"] == "transitional":
            insights["insights"].append("Predominantly transitional flow indicates a market changing character, potentially preparing for a regime shift")
        
        # Check for regime transitions
        if laminar.get('regions') and turbulent.get('regions'):
            insights["insights"].append("Coexistence of laminar and turbulent regions suggests market segmentation with different behaviors")
        
        return insights
    
    def _insights_from_pressure_gradients(self, pressure_gradients):
        """Generate insights from pressure gradient patterns"""
        if not pressure_gradients:
            return {"message": "No pressure gradient information available"}
        
        regions = pressure_gradients.get('regions', [])
        
        insights = {
            "count": len(regions),
            "average_magnitude": np.mean([r['magnitude'] for r in regions]) if regions else 0,
            "coverage": pressure_gradients.get('coverage', 0) * 100,
            "insights": []
        }
        
        # Generate specific insights
        if not regions:
            insights["insights"].append("No significant pressure gradients detected, suggesting equilibrium between buying and selling pressure")
            return insights
        
        # Analyze gradient directions
        directions = [r['direction'] for r in regions]
        upward = sum(1 for d in directions if -np.pi/4 < d < np.pi/4)
        downward = sum(1 for d in directions if d > 3*np.pi/4 or d < -3*np.pi/4)
        rightward = sum(1 for d in directions if np.pi/4 < d < 3*np.pi/4)
        leftward = sum(1 for d in directions if -3*np.pi/4 < d < -np.pi/4)
        
        if upward > downward and upward > rightward and upward > leftward:
            insights["insights"].append("Predominantly upward pressure gradients suggest building buying pressure")
        elif downward > upward and downward > rightward and downward > leftward:
            insights["insights"].append("Predominantly downward pressure gradients suggest building selling pressure")
        
        # Check for strong gradients
        strong_gradients = [r for r in regions if r['magnitude'] > 0.8]
        if strong_gradients:
            insights["insights"].append(f"Found {len(strong_gradients)} strong pressure gradients indicating potential for large price moves")
        
        return insights
    
    def _insights_from_shocks(self, shock_waves):
        """Generate insights from shock wave patterns"""
        if not shock_waves:
            return {"message": "No shock wave information available"}
        
        regions = shock_waves.get('regions', [])
        
        insights = {
            "count": len(regions),
            "average_strength": np.mean([r['strength'] for r in regions]) if regions else 0,
            "coverage": shock_waves.get('coverage', 0) * 100,
            "insights": []
        }
        
        # Generate specific insights
        if not regions:
            insights["insights"].append("No shock waves detected, suggesting smooth market activity without discontinuities")
            return insights
        
        if insights["count"] == 1:
            insights["insights"].append("Single shock wave detected, indicating a sudden market event causing discontinuity")
        elif insights["count"] > 1:
            insights["insights"].append(f"Multiple shock waves ({insights['count']}) detected, suggesting a highly reactive market with multiple discontinuities")
        
        if insights["average_strength"] > 0.8:
            insights["insights"].append("Very strong shock waves indicate significant market disruption, possibly due to major news or large orders")
        
        # Check for clustered shocks
        if insights["count"] > 2:
            # Calculate distances between shock centers
            centers = [r['center'] for r in regions]
            distances = []
            for i in range(len(centers)):
                for j in range(i+1, len(centers)):
                    y1, x1 = centers[i]
                    y2, x2 = centers[j]
                    distance = np.sqrt((y2-y1)**2 + (x2-x1)**2)
                    distances.append(distance)
            
            avg_distance = np.mean(distances) if distances else 0
            if avg_distance < 10:  # This threshold depends on field dimensions
                insights["insights"].append("Clustered shock waves suggest concentrated market activity in a specific price-time region")
        
        return insights
    
    def _insights_from_boundaries(self, boundary_layers):
        """Generate insights from boundary layer patterns"""
        if not boundary_layers:
            return {"message": "No boundary layer information available"}
        
        regions = boundary_layers.get('regions', [])
        
        insights = {
            "count": len(regions),
            "average_strength": np.mean([r['strength'] for r in regions]) if regions else 0,
            "average_length": np.mean([r['length'] for r in regions]) if regions else 0,
            "coverage": boundary_layers.get('coverage', 0) * 100,
            "insights": []
        }
        
        # Generate specific insights
        if not regions:
            insights["insights"].append("No significant boundary layers detected, suggesting absence of strong support/resistance levels")
            return insights
        
        if insights["count"] == 1:
            insights["insights"].append("Single major boundary layer detected, indicating a key support/resistance level")
        elif insights["count"] > 1:
            insights["insights"].append(f"Multiple boundary layers ({insights['count']}) detected, suggesting a market with several support/resistance levels")
        
        # Analyze boundary orientations
        orientations = [r['orientation'] for r in regions]
        horizontal = sum(1 for o in orientations if -np.pi/8 < (o % np.pi) < np.pi/8 or (o % np.pi) > 7*np.pi/8)
        vertical = sum(1 for o in orientations if np.pi/3 < (o % np.pi) < 2*np.pi/3)
        
        if horizontal > vertical:
            insights["insights"].append("Predominantly horizontal boundary layers indicate strong price-based support/resistance levels")
        elif vertical > horizontal:
            insights["insights"].append("Predominantly vertical boundary layers indicate time-based market structures")
        
        # Check for long boundaries
        long_boundaries = [r for r in regions if r['length'] > 20]  # This threshold depends on field dimensions
        if long_boundaries:
            insights["insights"].append(f"Found {len(long_boundaries)} extended boundary layers indicating persistent support/resistance zones")
        
        return insights
    
    def _assess_market_state(self, patterns, insights):
        """Generate overall market state assessment"""
        market_state = {
            "overall_regime": "unknown",
            "volatility": "unknown",
            "momentum": "unknown",
            "structure": "unknown",
            "key_points": []
        }
        
        # Determine overall regime
        flow_insights = insights.get('flow_regime_insights', {})
        dominant_regime = flow_insights.get('dominant_regime', 'none')
        
        if dominant_regime == "laminar":
            market_state["overall_regime"] = "trending"
        elif dominant_regime == "turbulent":
            market_state["overall_regime"] = "choppy"
        elif dominant_regime == "transitional":
            market_state["overall_regime"] = "transitional"
        
        # Assess volatility
        vortices = patterns.get('vortices', [])
        shock_waves = patterns.get('shock_waves', {}).get('regions', [])
        
        if len(vortices) > 3 or len(shock_waves) > 1:
            market_state["volatility"] = "high"
        elif len(vortices) > 1 or len(shock_waves) > 0:
            market_state["volatility"] = "medium"
        else:
            market_state["volatility"] = "low"
        
        # Assess momentum
        pressure_insights = insights.get('pressure_insights', {})
        pressure_regions = patterns.get('pressure_gradients', {}).get('regions', [])
        
        if pressure_regions:
            avg_magnitude = np.mean([r['magnitude'] for r in pressure_regions])
            if avg_magnitude > 0.7:
                market_state["momentum"] = "strong"
            elif avg_magnitude > 0.4:
                market_state["momentum"] = "moderate"
            else:
                market_state["momentum"] = "weak"
        
        # Assess market structure
        boundary_insights = insights.get('boundary_insights', {})
        boundary_regions = patterns.get('boundary_layers', {}).get('regions', [])
        
        if len(boundary_regions) > 3:
            market_state["structure"] = "complex"
        elif len(boundary_regions) > 1:
            market_state["structure"] = "structured"
        else:
            market_state["structure"] = "simple"
        
        # Identify key market points
        # Strong vortices as potential reversal points
        strong_vortices = [v for v in vortices if v['strength'] > 0.8]
        for v in strong_vortices:
            point = {
                "type": "reversal_point",
                "position": v['center'],
                "strength": v['strength'],
                "direction": "up" if v['sign'] > 0 else "down"
            }
            market_state["key_points"].append(point)
        
        # Strong boundaries as support/resistance
        strong_boundaries = [b for b in boundary_regions if b['strength'] > 0.7]
        for b in strong_boundaries:
            point = {
                "type": "support_resistance",
                "position": b['center'],
                "orientation": b['orientation'],
                "length": b['length']
            }
            market_state["key_points"].append(point)
        
        return market_state
    
    def visualize_patterns(self, flow_field, vorticity_field, pressure_field, patterns=None):
        """
        Visualize the identified patterns.
        
        Parameters:
        -----------
        flow_field : numpy.ndarray
            2D array of flow values
        vorticity_field : numpy.ndarray
            2D array of vorticity values
        pressure_field : numpy.ndarray
            2D array of pressure values
        patterns : dict, optional
            Dictionary of identified patterns. If None, uses self.patterns
        """
        if patterns is None:
            patterns = self.patterns
        
        if not patterns:
            print("No patterns to visualize. Run identify_patterns first.")
            return
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # 1. Flow field with vortices
        self._plot_flow_and_vortices(axes[0, 0], flow_field, vorticity_field, patterns)
        
        # 2. Flow regimes (laminar vs turbulent)
        self._plot_flow_regimes(axes[0, 1], flow_field, patterns)
        
        # 3. Pressure field with gradients
        self._plot_pressure_gradients(axes[1, 0], pressure_field, patterns)
        
        # 4. Composite view with shock waves and boundary layers
        self._plot_composite(axes[1, 1], flow_field, vorticity_field, pressure_field, patterns)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_flow_and_vortices(self, ax, flow_field, vorticity_field, patterns):
        """Helper method to plot flow field"""
        # Create meshgrid
        y, x = np.mgrid[0:flow_field.shape[0], 0:flow_field.shape[1]]
        
        # Calculate gradient for vector direction
        flow_dy, flow_dx = np.gradient(flow_field)
        
        # Plot vector field
        step = max(1, min(flow_field.shape) // 15)  # Downsample for clarity
        ax.quiver(x[::step, ::step], y[::step, ::step], 
                 flow_dx[::step, ::step], flow_dy[::step, ::step],
                 np.abs(flow_field[::step, ::step]), 
                 cmap='viridis', scale=30)
        
        # Add vortices
        vortices = patterns.get('vortices', [])
        patches = []
        
        for vortex in vortices:
            y_pos, x_pos = vortex['center']
            radius = np.sqrt(vortex['size'] / np.pi)  # Approximate radius from area
            circle = Circle((x_pos, y_pos), radius, alpha=0.3)
            patches.append(circle)
            
            # Add arrow to show rotation direction
            if vortex['direction'] == 'clockwise':
                ax.annotate('', xy=(x_pos+radius/2, y_pos), xytext=(x_pos, y_pos+radius/2),
                            arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
            else:
                ax.annotate('', xy=(x_pos, y_pos+radius/2), xytext=(x_pos+radius/2, y_pos),
                            arrowprops=dict(arrowstyle='->', color='white', lw=1.5))
        
        # Add vortex patches
        if patches:
            colors = np.linspace(0, 1, len(patches))
            collection = PatchCollection(patches, cmap=plt.cm.cool, alpha=0.3)
            collection.set_array(np.array(colors))
            ax.add_collection(collection)
        
        ax.set_title('Flow Field with Vortices')
        ax.set_xlabel('Price Level')
        ax.set_ylabel('Time')
    
    def _plot_flow_regimes(self, ax, flow_field, patterns):
        """Plot flow regimes (laminar vs turbulent)"""
        # Get flow regimes
        flow_regimes = patterns.get('flow_regimes', {})
        
        # Create visualization array (0 = no flow, 1 = laminar, 2 = transitional, 3 = turbulent)
        regimes_vis = np.zeros(flow_field.shape)
        
        if 'laminar' in flow_regimes:
            regimes_vis[flow_regimes['laminar']['mask']] = 1
        
        if 'transitional' in flow_regimes:
            regimes_vis[flow_regimes['transitional']['mask']] = 2
        
        if 'turbulent' in flow_regimes:
            regimes_vis[flow_regimes['turbulent']['mask']] = 3
        
        # Plot regimes
        cmap = plt.cm.get_cmap('viridis', 4)
        im = ax.imshow(regimes_vis, cmap=cmap, interpolation='nearest', origin='lower')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
        cbar.set_ticklabels(['No flow', 'Laminar', 'Transitional', 'Turbulent'])
        
        # Add labels for significant regions
        if 'laminar' in flow_regimes and flow_regimes['laminar'].get('regions'):
            for i, region in enumerate(flow_regimes['laminar']['regions']):
                y, x = region['center']
                ax.text(x, y, f"L{i+1}", color='white', fontweight='bold', ha='center', va='center')
        
        if 'turbulent' in flow_regimes and flow_regimes['turbulent'].get('regions'):
            for i, region in enumerate(flow_regimes['turbulent']['regions']):
                y, x = region['center']
                ax.text(x, y, f"T{i+1}", color='white', fontweight='bold', ha='center', va='center')
        
        ax.set_title('Flow Regimes')
        ax.set_xlabel('Price Level')
        ax.set_ylabel('Time')
    
    def _plot_pressure_gradients(self, ax, pressure_field, patterns):
        """Plot pressure field with gradients"""
        # Plot pressure field
        im = ax.imshow(pressure_field, cmap='coolwarm', origin='lower')
        plt.colorbar(im, ax=ax, label='Pressure')
        
        # Get pressure gradients
        pressure_gradients = patterns.get('pressure_gradients', {})
        
        if 'field' in pressure_gradients:
            # Get gradient field
            grad_y, grad_x = pressure_gradients['field']
            
            # Create meshgrid for quiver plot
            y, x = np.mgrid[0:pressure_field.shape[0], 0:pressure_field.shape[1]]
            
            # Subsample for clarity
            step = max(1, min(pressure_field.shape) // 20)
            
            # Plot gradient vectors
            ax.quiver(x[::step, ::step], y[::step, ::step], 
                     grad_x[::step, ::step], grad_y[::step, ::step],
                     color='black', scale=20, width=0.003)
        
        # Highlight significant gradient regions
        if 'regions' in pressure_gradients:
            for i, region in enumerate(pressure_gradients['regions']):
                y, x = region['center']
                ax.plot(x, y, 'o', color='lime', markersize=10*region['magnitude'])
                ax.text(x, y, f"G{i+1}", color='white', fontweight='bold', ha='center', va='center')
        
        ax.set_title('Pressure Field with Gradients')
        ax.set_xlabel('Price Level')
        ax.set_ylabel('Time')
    
    def _plot_composite(self, ax, flow_field, vorticity_field, pressure_field, patterns):
        """Plot composite view with shock waves and boundary layers"""
        # Background: flow magnitude
        flow_magnitude = np.abs(flow_field)
        im = ax.imshow(flow_magnitude, cmap='Greys', origin='lower', alpha=0.5)
        
        # Add shock waves
        shock_waves = patterns.get('shock_waves', {})
        if 'regions' in shock_waves:
            shock_patches = []
            for region in shock_waves['regions']:
                y, x = region['center']
                size = max(3, region['size'])
                rect = Rectangle((x-size/2, y-size/2), size, size, alpha=0.7)
                shock_patches.append(rect)
            
            shock_collection = PatchCollection(shock_patches, color='red', alpha=0.5)
            ax.add_collection(shock_collection)
            
            for i, region in enumerate(shock_waves['regions']):
                y, x = region['center']
                ax.text(x, y, f"S{i+1}", color='white', fontweight='bold', ha='center', va='center')
        
        # Add boundary layers
        boundary_layers = patterns.get('boundary_layers', {})
        if 'regions' in boundary_layers:
            for region in boundary_layers['regions']:
                points = region['points']
                y_coords, x_coords = zip(*points)
                ax.plot(x_coords, y_coords, '-', color='cyan', linewidth=2)
        
        # Add key market points from overall assessment
        market_state = patterns.get('market_state', {})
        key_points = market_state.get('key_points', [])
        
        for i, point in enumerate(key_points):
            y, x = point['position']
            
            if point['type'] == 'reversal_point':
                color = 'lime' if point['direction'] == 'up' else 'magenta'
                ax.plot(x, y, 'o', color=color, markersize=10)
                ax.text(x, y+2, f"R{i+1}", color='white', fontweight='bold', ha='center')
            
            elif point['type'] == 'support_resistance':
                ax.plot(x, y, 's', color='yellow', markersize=8)
                ax.text(x, y+2, f"SR{i+1}", color='white', fontweight='bold', ha='center')
        
        ax.set_title('Composite Market Structure')
        ax.set_xlabel('Price Level')
        ax.set_ylabel('Time')

    def _plot_flow_field(self, ax, flow_field, vorticity_field, patterns):
        """Helper method to plot flow field"""
        # Create meshgrid
        y, x = np.mgrid[0:flow_field.shape[0], 0:flow_field.shape[1]]
        
        # Calculate flow gradient for visualization
        dy, dx = np.gradient(flow_field)
        
        # Plot quiver instead of streamplot to avoid unequal rows issue
        step = max(1, min(flow_field.shape) // 15)  # Downsample for clarity
        ax.quiver(x[::step, ::step], y[::step, ::step], 
                 dx[::step, ::step], dy[::step, ::step],
                 np.abs(flow_field[::step, ::step]), 
                 cmap='viridis', scale=30)
        
        ax.set_title('Flow Field')
        ax.set_xlabel('Price Level')
        ax.set_ylabel('Time')


# Example usage
def test_pattern_recognizer():
    """Test the pattern recognizer with synthetic data"""
    # Create synthetic fields
    size = 50
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    
    # Create flow field (with a vortex and a laminar region)
    cx, cy = 0.3, 0.3  # Vortex center
    r = np.sqrt((x - cx)**2 + (y - cy)**2)
    theta = np.arctan2(y - cy, x - cx)
    
    # Vortex component
    vortex_x = -np.sin(theta) * np.exp(-10*r)
    vortex_y = np.cos(theta) * np.exp(-10*r)
    
    # Laminar component
    laminar_x = 0.5 * (x > 0.5)
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
    pressure_field = np.exp(-5*((x-0.7)**2 + (y-0.7)**2)) - np.exp(-10*((x-0.2)**2 + (y-0.5)**2))
    
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