#!/usr/bin/env python3
"""
Comprehensive demo script for the Fluid Dynamics HFT System.

This script demonstrates the key capabilities of the FluidTrade system:
- Market data generation and processing
- Fluid dynamics model computation
- Pattern recognition and analysis
- Trading signal generation
- Strategy execution and performance tracking
- Visualization and reporting

Usage:
    python demo_system.py
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import our modules
from fluid_market_model import FluidMarketModel
from fluid_hft_strategy import FluidHFTStrategy
from fluid_market_system import FluidMarketSystem
from pattern_identification import FluidPatternRecognizer
from utils.data_loaders import load_market_data
from utils.visualization import FluidDynamicsVisualizer, create_summary_report


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('fluid_hft_demo.log')
        ]
    )


def demo_data_generation():
    """Demonstrate data generation capabilities."""
    print("\n" + "="*60)
    print("1. DATA GENERATION DEMO")
    print("="*60)
    
    # Generate synthetic market data
    print("Generating synthetic market data...")
    data = load_market_data(ticker="DEMO", n_points=5000)
    
    print(f"Generated {len(data)} market data points")
    print(f"Price range: ${data['Price'].min():.2f} - ${data['Price'].max():.2f}")
    print(f"Volume range: {data['Size'].min()} - {data['Size'].max()}")
    print(f"Time range: {data['Time'].min()} to {data['Time'].max()}")
    
    return data


def demo_fluid_model(data):
    """Demonstrate fluid dynamics model capabilities."""
    print("\n" + "="*60)
    print("2. FLUID DYNAMICS MODEL DEMO")
    print("="*60)
    
    # Initialize model
    model_config = {
        'viscosity': 0.01,
        'density': 1.0,
        'time_step': 0.01,
        'time_window': 100,
        'price_levels': 50
    }
    
    model = FluidMarketModel(config=model_config)
    model.data = data
    
    print("Computing fluid dynamics fields...")
    
    # Compute all fields
    flow_field = model.compute_flow_field()
    pressure_field = model.compute_pressure_field()
    vorticity_field = model.compute_vorticity_field()
    
    print(f"Flow field shape: {flow_field.shape}")
    print(f"Flow field range: {flow_field.min():.3f} to {flow_field.max():.3f}")
    print(f"Pressure field range: {pressure_field.min():.3f} to {pressure_field.max():.3f}")
    print(f"Vorticity field range: {vorticity_field.min():.3f} to {vorticity_field.max():.3f}")
    
    # Detect regime transitions
    transitions = model.detect_regime_transitions()
    print(f"\nRegime transitions detected:")
    for regime, indices in transitions.items():
        if indices:
            print(f"  {regime}: {len(indices)} transitions")
    
    return model


def demo_pattern_recognition(model):
    """Demonstrate pattern recognition capabilities."""
    print("\n" + "="*60)
    print("3. PATTERN RECOGNITION DEMO")
    print("="*60)
    
    # Initialize pattern recognizer
    recognizer = FluidPatternRecognizer()
    recognizer.initialize()
    
    print("Detecting fluid dynamics patterns...")
    
    # Detect patterns
    patterns = recognizer.detect_patterns(
        model.flow_field,
        model.pressure_field,
        model.vorticity_field
    )
    
    print(f"Patterns detected:")
    for pattern_type, pattern_data in patterns.items():
        if pattern_type == 'vortices':
            print(f"  Vortices: {len(pattern_data)} detected")
        elif pattern_type in ['laminar', 'turbulent']:
            coverage = pattern_data.get('coverage', 0)
            print(f"  {pattern_type.capitalize()} flow: {coverage:.1%} coverage")
        elif pattern_type == 'pressure_gradients':
            num_regions = pattern_data.get('num_regions', 0)
            print(f"  Pressure gradients: {num_regions} regions")
    
    # Generate market insights
    insights = recognizer.generate_market_insights(patterns)
    print(f"\nMarket insights generated:")
    for category, category_insights in insights.items():
        if category == 'market_state':
            state = category_insights
            print(f"  Market regime: {state.get('regime', 'unknown')}")
            print(f"  Volatility: {state.get('volatility', 'unknown')}")
            print(f"  Momentum: {state.get('momentum', 'unknown')}")
    
    return patterns, insights


def demo_trading_strategy(model, patterns):
    """Demonstrate trading strategy capabilities."""
    print("\n" + "="*60)
    print("4. TRADING STRATEGY DEMO")
    print("="*60)
    
    # Initialize strategy
    strategy_config = {
        'position_size_limit': 1000,
        'max_drawdown': 0.05,
        'risk_factor': 0.1,
        'min_confidence': 0.6,
        'flow_threshold': 0.3,
        'pressure_threshold': 0.2,
        'vorticity_threshold': 0.4
    }
    
    strategy = FluidHFTStrategy(model, config=strategy_config)
    
    print("Generating trading signals...")
    
    # Generate signals for multiple time points
    signals = []
    trades = []
    
    for i in range(10, min(90, model.time_window - 1)):
        signal = strategy.generate_signal(i)
        signals.append(signal)
        
        # Simulate trade execution
        if signal['signal'].name != 'HOLD':
            current_price = 100 + np.random.normal(0, 0.1)  # Simulate current price
            trade_time = datetime.now() + timedelta(seconds=i)
            
            trade = strategy.execute_trade(
                signal, current_price, "DEMO", trade_time
            )
            
            if trade:
                trades.append(trade)
    
    print(f"Generated {len(signals)} signals")
    print(f"Executed {len(trades)} trades")
    
    # Calculate performance
    performance = strategy.get_performance_metrics()
    print(f"\nPerformance metrics:")
    print(f"  Total PnL: ${performance.get('total_pnl', 0):,.2f}")
    print(f"  Total trades: {performance.get('total_trades', 0)}")
    print(f"  Buy trades: {performance.get('buy_trades', 0)}")
    print(f"  Sell trades: {performance.get('sell_trades', 0)}")
    print(f"  Total volume: ${performance.get('total_volume', 0):,.2f}")
    
    return strategy, signals, trades


def demo_system_orchestration():
    """Demonstrate full system orchestration."""
    print("\n" + "="*60)
    print("5. SYSTEM ORCHESTRATION DEMO")
    print("="*60)
    
    # Initialize system
    system_config = {
        'model': {
            'viscosity': 0.01,
            'time_window': 50,
            'price_levels': 30
        },
        'strategy': {
            'position_size_limit': 500,
            'risk_factor': 0.08
        },
        'patterns': {
            'vorticity_threshold': 0.5
        },
        'update_frequency': 0.1,
        'data_buffer_size': 5000
    }
    
    system = FluidMarketSystem(config=system_config)
    
    print("Starting system...")
    system.start()
    
    # Let system run for a short time
    import time
    time.sleep(2)
    
    # Get system status
    status = system.get_system_status()
    print(f"\nSystem status:")
    print(f"  State: {status['state']}")
    print(f"  Uptime: {status['uptime']:.1f} seconds")
    print(f"  Data points processed: {status['data_points']}")
    print(f"  Signals generated: {status['signals_generated']}")
    print(f"  Trades executed: {status['trades_executed']}")
    
    # Get recent activity
    recent_signals = system.get_recent_signals(limit=3)
    recent_trades = system.get_recent_trades(limit=3)
    
    print(f"\nRecent signals: {len(recent_signals)}")
    print(f"Recent trades: {len(recent_trades)}")
    
    # Stop system
    system.stop()
    print("System stopped.")
    
    return system


def demo_visualization(model, patterns, strategy, data):
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("6. VISUALIZATION DEMO")
    print("="*60)
    
    # Initialize visualizer
    visualizer = FluidDynamicsVisualizer()
    
    print("Creating visualizations...")
    
    # Create fluid dynamics visualization
    fig1 = visualizer.plot_fluid_fields(
        model.flow_field,
        model.pressure_field,
        model.vorticity_field,
        patterns,
        title="Fluid Dynamics Analysis"
    )
    
    # Create market data overview
    fig2 = visualizer.plot_market_data_overview(
        data,
        title="Market Data Overview"
    )
    
    # Create performance metrics visualization
    performance = strategy.get_performance_metrics()
    fig3 = visualizer.plot_performance_metrics(
        performance,
        title="Trading Performance"
    )
    
    # Save visualizations
    fig1.savefig('fluid_dynamics_fields.png', dpi=300, bbox_inches='tight')
    fig2.savefig('market_data_overview.png', dpi=300, bbox_inches='tight')
    fig3.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    
    print("Visualizations saved:")
    print("  - fluid_dynamics_fields.png")
    print("  - market_data_overview.png")
    print("  - performance_metrics.png")
    
    return fig1, fig2, fig3


def demo_backtesting():
    """Demonstrate backtesting capabilities."""
    print("\n" + "="*60)
    print("7. BACKTESTING DEMO")
    print("="*60)
    
    # Initialize system for backtesting
    system = FluidMarketSystem()
    
    print("Running backtest...")
    
    # Run backtest
    backtest_results = system.run_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0
    )
    
    print(f"Backtest results:")
    print(f"  Initial capital: ${backtest_results['initial_capital']:,.2f}")
    print(f"  Final capital: ${backtest_results['final_capital']:,.2f}")
    print(f"  Total return: {backtest_results['total_return']:.2%}")
    print(f"  Sharpe ratio: {backtest_results['sharpe_ratio']:.3f}")
    print(f"  Max drawdown: {backtest_results['max_drawdown']:.2%}")
    
    return backtest_results


def create_demo_report(data, model, patterns, strategy, system, backtest_results):
    """Create a comprehensive demo report."""
    print("\n" + "="*60)
    print("8. DEMO REPORT GENERATION")
    print("="*60)
    
    # Prepare data for report
    system_status = system.get_system_status() if system else {'state': 'stopped'}
    recent_trades = system.get_recent_trades() if system else []
    recent_signals = system.get_recent_signals() if system else []
    performance_metrics = strategy.get_performance_metrics()
    
    # Generate report
    report = create_summary_report(
        system_status,
        recent_trades,
        recent_signals,
        performance_metrics
    )
    
    # Add additional sections
    report += "\n\n" + "=" * 60
    report += "\nFLUID DYNAMICS ANALYSIS SUMMARY"
    report += "\n" + "=" * 60
    
    report += f"\n\n📊 DATA STATISTICS:"
    report += f"\n  Data points: {len(data):,}"
    report += f"\n  Price range: ${data['Price'].min():.2f} - ${data['Price'].max():.2f}"
    report += f"\n  Volume range: {data['Size'].min():,} - {data['Size'].max():,}"
    
    report += f"\n\n🌊 FLUID DYNAMICS FIELDS:"
    report += f"\n  Flow field range: {model.flow_field.min():.3f} to {model.flow_field.max():.3f}"
    report += f"\n  Pressure field range: {model.pressure_field.min():.3f} to {model.pressure_field.max():.3f}"
    report += f"\n  Vorticity field range: {model.vorticity_field.min():.3f} to {model.vorticity_field.max():.3f}"
    
    report += f"\n\n🔍 PATTERN DETECTION:"
    vortices = patterns.get('vortices', [])
    laminar = patterns.get('laminar', {})
    turbulent = patterns.get('turbulent', {})
    
    report += f"\n  Vortices detected: {len(vortices)}"
    report += f"\n  Laminar flow coverage: {laminar.get('coverage', 0):.1%}"
    report += f"\n  Turbulent flow coverage: {turbulent.get('coverage', 0):.1%}"
    
    report += f"\n\n📈 BACKTESTING RESULTS:"
    report += f"\n  Total return: {backtest_results['total_return']:.2%}"
    report += f"\n  Sharpe ratio: {backtest_results['sharpe_ratio']:.3f}"
    report += f"\n  Max drawdown: {backtest_results['max_drawdown']:.2%}"
    
    report += "\n\n" + "=" * 60
    report += "\nDEMO COMPLETED SUCCESSFULLY!"
    report += "\n" + "=" * 60
    
    # Save report
    with open('fluid_hft_demo_report.txt', 'w') as f:
        f.write(report)
    
    print("Demo report saved to: fluid_hft_demo_report.txt")
    print(report)


def main():
    """Main demo function."""
    print("FluidTrade: Fluid Dynamics High-Frequency Trading System")
    print("Comprehensive System Demonstration")
    print("=" * 60)
    
    # Setup logging
    setup_logging()
    
    try:
        # Run all demos
        data = demo_data_generation()
        model = demo_fluid_model(data)
        patterns, insights = demo_pattern_recognition(model)
        strategy, signals, trades = demo_trading_strategy(model, patterns)
        system = demo_system_orchestration()
        visualizations = demo_visualization(model, patterns, strategy, data)
        backtest_results = demo_backtesting()
        
        # Generate comprehensive report
        create_demo_report(data, model, patterns, strategy, system, backtest_results)
        
        print("\n" + "="*60)
        print("✅ ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nGenerated files:")
        print("  - fluid_hft_demo.log (system log)")
        print("  - fluid_hft_demo_report.txt (comprehensive report)")
        print("  - fluid_dynamics_fields.png (fluid dynamics visualization)")
        print("  - market_data_overview.png (market data analysis)")
        print("  - performance_metrics.png (trading performance)")
        print("\nThe FluidTrade system is now fully operational and ready for use!")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)