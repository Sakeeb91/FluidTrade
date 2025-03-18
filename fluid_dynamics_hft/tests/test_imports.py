import unittest

class TestImports(unittest.TestCase):
    """Test that all modules import correctly."""
    
    def test_import_main_modules(self):
        """Test importing main modules."""
        try:
            from fluid_dynamics_hft.src.fluid_hft_strategy import FluidHFTStrategy
            from fluid_dynamics_hft.src.fluid_market_model import FluidMarketModel
            from fluid_dynamics_hft.src.fluid_market_system import FluidMarketSystem
            from fluid_dynamics_hft.src.pattern_identification import FluidPatternRecognizer
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import main modules: {e}")
            
    def test_import_utils(self):
        """Test importing utility modules."""
        try:
            from fluid_dynamics_hft.src.utils.data_loaders import load_lobster_data
            from fluid_dynamics_hft.src.utils.visualization import (
                plot_flow_field, plot_vorticity_field, plot_pressure_field, 
                plot_pattern_overlay
            )
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import utility modules: {e}")
            
    def test_strategy_instantiation(self):
        """Test that the strategy class can be instantiated."""
        try:
            from fluid_dynamics_hft.src.fluid_hft_strategy import FluidHFTStrategy
            strategy = FluidHFTStrategy()
            self.assertIsNotNone(strategy)
        except Exception as e:
            self.fail(f"Failed to instantiate FluidHFTStrategy: {e}")
            
    def test_model_instantiation(self):
        """Test that the model class can be instantiated."""
        try:
            from fluid_dynamics_hft.src.fluid_market_model import FluidMarketModel
            model = FluidMarketModel()
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Failed to instantiate FluidMarketModel: {e}")
            
    def test_system_instantiation(self):
        """Test that the system class can be instantiated."""
        try:
            from fluid_dynamics_hft.src.fluid_market_system import FluidMarketSystem
            system = FluidMarketSystem()
            self.assertIsNotNone(system)
        except Exception as e:
            self.fail(f"Failed to instantiate FluidMarketSystem: {e}")
            
    def test_pattern_recognizer_instantiation(self):
        """Test that the pattern recognizer class can be instantiated."""
        try:
            from fluid_dynamics_hft.src.pattern_identification import FluidPatternRecognizer
            recognizer = FluidPatternRecognizer()
            self.assertIsNotNone(recognizer)
        except Exception as e:
            self.fail(f"Failed to instantiate FluidPatternRecognizer: {e}")


if __name__ == '__main__':
    unittest.main() 