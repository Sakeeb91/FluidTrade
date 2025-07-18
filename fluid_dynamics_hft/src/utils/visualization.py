"""
Visualization utilities for the fluid dynamics HFT system.

This module provides comprehensive visualization tools for market data, fluid dynamics fields,
trading signals, and performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import warnings
warnings.filterwarnings('ignore')


class FluidDynamicsVisualizer:
    """
    Visualizer for fluid dynamics market analysis.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.style = self.config.get('style', 'seaborn-v0_8-darkgrid')
        self.figsize = self.config.get('figsize', (12, 8))
        self.colors = self.config.get('colors', ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        
        # Set matplotlib style
        plt.style.use(self.style)
        sns.set_palette(self.colors)
    
    def plot_fluid_fields(self, flow_field: np.ndarray, pressure_field: np.ndarray, 
                         vorticity_field: np.ndarray, patterns: Optional[Dict] = None,
                         title: str = "Fluid Dynamics Fields") -> plt.Figure:
        """
        Plot the three main fluid dynamics fields.
        
        Args:
            flow_field: 2D array of flow magnitudes
            pressure_field: 2D array of pressure values
            vorticity_field: 2D array of vorticity values
            patterns: Optional dictionary of detected patterns
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot flow field
        im1 = axes[0, 0].imshow(flow_field, cmap='viridis', aspect='auto', origin='lower')
        axes[0, 0].set_title('Flow Field (Magnitude)', fontweight='bold')
        axes[0, 0].set_xlabel('Price Levels')
        axes[0, 0].set_ylabel('Time Steps')
        plt.colorbar(im1, ax=axes[0, 0], label='Flow Magnitude')
        
        # Plot pressure field
        im2 = axes[0, 1].imshow(pressure_field, cmap='RdBu_r', aspect='auto', origin='lower')
        axes[0, 1].set_title('Pressure Field', fontweight='bold')
        axes[0, 1].set_xlabel('Price Levels')
        axes[0, 1].set_ylabel('Time Steps')
        plt.colorbar(im2, ax=axes[0, 1], label='Pressure')
        
        # Plot vorticity field
        im3 = axes[1, 0].imshow(vorticity_field, cmap='RdBu', aspect='auto', origin='lower')
        axes[1, 0].set_title('Vorticity Field', fontweight='bold')
        axes[1, 0].set_xlabel('Price Levels')
        axes[1, 0].set_ylabel('Time Steps')
        plt.colorbar(im3, ax=axes[1, 0], label='Vorticity')
        
        # Highlight detected patterns
        if patterns and 'vortices' in patterns:
            for vortex in patterns['vortices']:
                center = vortex['center']
                circle = plt.Circle((center[1], center[0]), radius=3, 
                                  fill=False, color='yellow', linewidth=2)
                axes[1, 0].add_patch(circle)
        
        # Plot combined regime map
        regime_map = self._create_regime_map(flow_field, pressure_field, vorticity_field, patterns)
        im4 = axes[1, 1].imshow(regime_map, cmap='tab10', aspect='auto', origin='lower')
        axes[1, 1].set_title('Market Regimes', fontweight='bold')
        axes[1, 1].set_xlabel('Price Levels')
        axes[1, 1].set_ylabel('Time Steps')
        
        # Add legend for regimes
        regime_labels = ['Neutral', 'Laminar', 'Turbulent', 'High Pressure', 'Vortex']
        colors = plt.cm.tab10(np.linspace(0, 1, len(regime_labels)))
        legend_elements = [patches.Patch(color=colors[i], label=regime_labels[i]) 
                          for i in range(len(regime_labels))]
        axes[1, 1].legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        
        plt.tight_layout()
        return fig
    
    def _create_regime_map(self, flow_field: np.ndarray, pressure_field: np.ndarray, 
                          vorticity_field: np.ndarray, patterns: Optional[Dict] = None) -> np.ndarray:
        """Create a combined regime map from all fields."""
        regime_map = np.zeros_like(flow_field)
        
        # Define thresholds
        flow_threshold = np.percentile(np.abs(flow_field), 70)
        pressure_threshold = np.percentile(np.abs(pressure_field), 80)
        vorticity_threshold = np.percentile(np.abs(vorticity_field), 75)
        
        # Assign regimes
        laminar_mask = (np.abs(flow_field) > flow_threshold) & (np.abs(vorticity_field) < vorticity_threshold/2)
        turbulent_mask = np.abs(vorticity_field) > vorticity_threshold
        pressure_mask = np.abs(pressure_field) > pressure_threshold
        
        regime_map[laminar_mask] = 1
        regime_map[turbulent_mask] = 2
        regime_map[pressure_mask] = 3
        
        # Add vortex locations
        if patterns and 'vortices' in patterns:
            for vortex in patterns['vortices']:
                center = vortex['center']
                y, x = int(center[0]), int(center[1])
                size = max(2, int(vortex['size'] ** 0.5))
                y_min, y_max = max(0, y-size), min(regime_map.shape[0], y+size)
                x_min, x_max = max(0, x-size), min(regime_map.shape[1], x+size)
                regime_map[y_min:y_max, x_min:x_max] = 4
        
        return regime_map
    
    def plot_market_data_overview(self, data: pd.DataFrame, title: str = "Market Data Overview") -> plt.Figure:
        """
        Plot comprehensive market data overview.
        
        Args:
            data: Market data DataFrame
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Price time series
        axes[0, 0].plot(data['Time'], data['Price'], linewidth=1, alpha=0.8)
        axes[0, 0].set_title('Price Evolution', fontweight='bold')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Price')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Volume time series
        axes[0, 1].plot(data['Time'], data['Size'], linewidth=1, alpha=0.8, color='orange')
        axes[0, 1].set_title('Volume Evolution', fontweight='bold')
        axes[0, 1].set_xlabel('Time')
        axes[0, 1].set_ylabel('Volume')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Price distribution
        axes[1, 0].hist(data['Price'], bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Price Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Price')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Volume distribution
        axes[1, 1].hist(data['Size'], bins=50, alpha=0.7, color='red')
        axes[1, 1].set_title('Volume Distribution', fontweight='bold')
        axes[1, 1].set_xlabel('Volume')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Returns if available
        if 'Returns' in data.columns:
            axes[2, 0].plot(data['Time'], data['Returns'], linewidth=1, alpha=0.8, color='purple')
            axes[2, 0].set_title('Returns', fontweight='bold')
            axes[2, 0].set_xlabel('Time')
            axes[2, 0].set_ylabel('Returns')
            axes[2, 0].grid(True, alpha=0.3)
            
            # Returns distribution
            axes[2, 1].hist(data['Returns'].dropna(), bins=50, alpha=0.7, color='brown')
            axes[2, 1].set_title('Returns Distribution', fontweight='bold')
            axes[2, 1].set_xlabel('Returns')
            axes[2, 1].set_ylabel('Frequency')
            axes[2, 1].grid(True, alpha=0.3)
        else:
            # Price vs Volume scatter
            axes[2, 0].scatter(data['Price'], data['Size'], alpha=0.5, s=1)
            axes[2, 0].set_title('Price vs Volume', fontweight='bold')
            axes[2, 0].set_xlabel('Price')
            axes[2, 0].set_ylabel('Volume')
            axes[2, 0].grid(True, alpha=0.3)
            
            # Order flow if available
            if 'Direction' in data.columns:
                buy_orders = data[data['Direction'] > 0]
                sell_orders = data[data['Direction'] < 0]
                
                axes[2, 1].hist([buy_orders['Size'], sell_orders['Size']], 
                               bins=30, alpha=0.7, label=['Buy Orders', 'Sell Orders'])
                axes[2, 1].set_title('Order Flow Distribution', fontweight='bold')
                axes[2, 1].set_xlabel('Order Size')
                axes[2, 1].set_ylabel('Frequency')
                axes[2, 1].legend()
                axes[2, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_trading_signals(self, signals: List[Dict], prices: pd.Series, 
                           title: str = "Trading Signals") -> plt.Figure:
        """
        Plot trading signals overlaid on price data.
        
        Args:
            signals: List of signal dictionaries
            prices: Price time series
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Plot price with signals
        axes[0].plot(prices.index, prices.values, linewidth=1, alpha=0.8, label='Price')
        
        # Plot signals
        for signal in signals:
            if 'timestamp' in signal and 'signal' in signal:
                timestamp = signal['timestamp']
                signal_type = signal['signal']
                confidence = signal.get('confidence', 0.5)
                
                if hasattr(signal_type, 'name'):
                    signal_name = signal_type.name
                else:
                    signal_name = str(signal_type)
                
                if signal_name == 'BUY':
                    axes[0].scatter(timestamp, prices.iloc[min(timestamp, len(prices)-1)], 
                                   color='green', marker='^', s=100*confidence, 
                                   alpha=0.7, label='Buy Signal' if timestamp == signals[0]['timestamp'] else '')
                elif signal_name == 'SELL':
                    axes[0].scatter(timestamp, prices.iloc[min(timestamp, len(prices)-1)], 
                                   color='red', marker='v', s=100*confidence, 
                                   alpha=0.7, label='Sell Signal' if timestamp == signals[0]['timestamp'] else '')
        
        axes[0].set_title('Price with Trading Signals', fontweight='bold')
        axes[0].set_xlabel('Time Index')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot signal confidence over time
        signal_times = [s['timestamp'] for s in signals if 'timestamp' in s]
        signal_confidences = [s.get('confidence', 0.5) for s in signals if 'timestamp' in s]
        
        if signal_times and signal_confidences:
            axes[1].plot(signal_times, signal_confidences, marker='o', linewidth=2, markersize=4)
            axes[1].set_title('Signal Confidence Over Time', fontweight='bold')
            axes[1].set_xlabel('Time Index')
            axes[1].set_ylabel('Confidence')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def plot_performance_metrics(self, metrics: Dict[str, Any], 
                               title: str = "Performance Metrics") -> plt.Figure:
        """
        Plot comprehensive performance metrics.
        
        Args:
            metrics: Dictionary of performance metrics
            title: Plot title
            
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Key metrics summary
        key_metrics = {
            'Total PnL': metrics.get('total_pnl', 0),
            'Realized PnL': metrics.get('realized_pnl', 0),
            'Unrealized PnL': metrics.get('unrealized_pnl', 0),
            'Total Trades': metrics.get('total_trades', 0),
            'Current Drawdown': metrics.get('current_drawdown', 0)
        }
        
        metric_names = list(key_metrics.keys())
        metric_values = list(key_metrics.values())
        
        axes[0, 0].bar(metric_names, metric_values, color=['green' if v >= 0 else 'red' for v in metric_values])
        axes[0, 0].set_title('Key Performance Metrics', fontweight='bold')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Trade statistics
        trade_stats = {
            'Buy Trades': metrics.get('buy_trades', 0),
            'Sell Trades': metrics.get('sell_trades', 0)
        }
        
        if sum(trade_stats.values()) > 0:
            axes[0, 1].pie(trade_stats.values(), labels=trade_stats.keys(), autopct='%1.1f%%')
            axes[0, 1].set_title('Trade Distribution', fontweight='bold')
        else:
            axes[0, 1].text(0.5, 0.5, 'No trades executed', ha='center', va='center', 
                           transform=axes[0, 1].transAxes, fontsize=14)
            axes[0, 1].set_title('Trade Distribution', fontweight='bold')
        
        # Volume and commission analysis
        volume_data = {
            'Total Volume': metrics.get('total_volume', 0),
            'Average Trade Size': metrics.get('avg_trade_size', 0),
            'Total Commission': metrics.get('total_commission', 0)
        }
        
        volume_names = list(volume_data.keys())
        volume_values = list(volume_data.values())
        
        axes[1, 0].bar(volume_names, volume_values, color='blue', alpha=0.7)
        axes[1, 0].set_title('Volume & Commission Analysis', fontweight='bold')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Risk metrics
        risk_metrics = {
            'Current Drawdown': metrics.get('current_drawdown', 0),
            'Win Rate': metrics.get('win_rate', 0)
        }
        
        # Create gauge-style plot for risk metrics
        risk_names = list(risk_metrics.keys())
        risk_values = list(risk_metrics.values())
        
        bars = axes[1, 1].bar(risk_names, risk_values, color=['red', 'green'], alpha=0.7)
        axes[1, 1].set_title('Risk Metrics', fontweight='bold')
        axes[1, 1].set_ylabel('Value')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, risk_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{value:.2%}', ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    def create_interactive_dashboard(self, data: pd.DataFrame, signals: List[Dict], 
                                   metrics: Dict[str, Any]) -> go.Figure:
        """
        Create an interactive dashboard using Plotly.
        
        Args:
            data: Market data
            signals: Trading signals
            metrics: Performance metrics
            
        Returns:
            Plotly Figure object
        """
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Price Evolution', 'Volume Evolution', 
                           'Signal Confidence', 'Performance Metrics',
                           'Price Distribution', 'Trade Distribution'),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Price evolution
        fig.add_trace(
            go.Scatter(x=data['Time'], y=data['Price'], mode='lines', name='Price', 
                      line=dict(color='blue', width=1)),
            row=1, col=1
        )
        
        # Volume evolution
        fig.add_trace(
            go.Scatter(x=data['Time'], y=data['Size'], mode='lines', name='Volume',
                      line=dict(color='orange', width=1)),
            row=1, col=2
        )
        
        # Signal confidence
        if signals:
            signal_times = [s['timestamp'] for s in signals if 'timestamp' in s]
            signal_confidences = [s.get('confidence', 0.5) for s in signals if 'timestamp' in s]
            
            fig.add_trace(
                go.Scatter(x=signal_times, y=signal_confidences, mode='lines+markers',
                          name='Signal Confidence', line=dict(color='green', width=2)),
                row=2, col=1
            )
        
        # Performance metrics
        metric_names = ['Total PnL', 'Realized PnL', 'Unrealized PnL', 'Total Trades']
        metric_values = [metrics.get(name.lower().replace(' ', '_'), 0) for name in metric_names]
        
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name='Performance',
                   marker_color=['green' if v >= 0 else 'red' for v in metric_values]),
            row=2, col=2
        )
        
        # Price distribution
        fig.add_trace(
            go.Histogram(x=data['Price'], nbinsx=50, name='Price Distribution',
                        marker_color='lightblue'),
            row=3, col=1
        )
        
        # Trade distribution
        if metrics.get('buy_trades', 0) > 0 or metrics.get('sell_trades', 0) > 0:
            fig.add_trace(
                go.Pie(labels=['Buy Trades', 'Sell Trades'], 
                      values=[metrics.get('buy_trades', 0), metrics.get('sell_trades', 0)],
                      name='Trade Distribution'),
                row=3, col=2
            )
        
        fig.update_layout(
            title_text="Fluid Dynamics HFT Dashboard",
            showlegend=True,
            height=900
        )
        
        return fig
    
    def animate_fluid_evolution(self, flow_field: np.ndarray, pressure_field: np.ndarray, 
                               vorticity_field: np.ndarray, save_path: Optional[str] = None) -> FuncAnimation:
        """
        Create an animation of fluid field evolution.
        
        Args:
            flow_field: 3D array (time, height, width) of flow values
            pressure_field: 3D array (time, height, width) of pressure values
            vorticity_field: 3D array (time, height, width) of vorticity values
            save_path: Optional path to save animation
            
        Returns:
            matplotlib FuncAnimation object
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Initialize plots
        im1 = axes[0].imshow(flow_field[0], cmap='viridis', aspect='auto', origin='lower')
        axes[0].set_title('Flow Field')
        plt.colorbar(im1, ax=axes[0])
        
        im2 = axes[1].imshow(pressure_field[0], cmap='RdBu_r', aspect='auto', origin='lower')
        axes[1].set_title('Pressure Field')
        plt.colorbar(im2, ax=axes[1])
        
        im3 = axes[2].imshow(vorticity_field[0], cmap='RdBu', aspect='auto', origin='lower')
        axes[2].set_title('Vorticity Field')
        plt.colorbar(im3, ax=axes[2])
        
        def animate(frame):
            im1.set_array(flow_field[frame])
            im2.set_array(pressure_field[frame])
            im3.set_array(vorticity_field[frame])
            fig.suptitle(f'Fluid Dynamics Evolution - Time Step {frame}', fontsize=16)
            return [im1, im2, im3]
        
        anim = FuncAnimation(fig, animate, frames=len(flow_field), interval=100, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        return anim


def create_summary_report(system_status: Dict, recent_trades: List[Dict], 
                         recent_signals: List[Dict], performance_metrics: Dict) -> str:
    """
    Create a text summary report of system status.
    
    Args:
        system_status: System status dictionary
        recent_trades: List of recent trades
        recent_signals: List of recent signals
        performance_metrics: Performance metrics dictionary
        
    Returns:
        Formatted summary report string
    """
    report = []
    report.append("=" * 60)
    report.append("FLUID DYNAMICS HFT SYSTEM SUMMARY REPORT")
    report.append("=" * 60)
    
    # System Status
    report.append("\n=� SYSTEM STATUS:")
    report.append(f"  State: {system_status.get('state', 'Unknown')}")
    report.append(f"  Uptime: {system_status.get('uptime', 0):.1f} seconds")
    report.append(f"  Data Points Processed: {system_status.get('data_points', 0):,}")
    report.append(f"  Signals Generated: {system_status.get('signals_generated', 0):,}")
    report.append(f"  Trades Executed: {system_status.get('trades_executed', 0):,}")
    
    # Performance Metrics
    report.append("\n=� PERFORMANCE METRICS:")
    report.append(f"  Total PnL: ${performance_metrics.get('total_pnl', 0):,.2f}")
    report.append(f"  Realized PnL: ${performance_metrics.get('realized_pnl', 0):,.2f}")
    report.append(f"  Unrealized PnL: ${performance_metrics.get('unrealized_pnl', 0):,.2f}")
    report.append(f"  Current Drawdown: {performance_metrics.get('current_drawdown', 0):.2%}")
    report.append(f"  Total Volume: ${performance_metrics.get('total_volume', 0):,.2f}")
    
    # Recent Trades
    report.append("\n📈 RECENT TRADES:")
    if recent_trades:
        for trade in recent_trades[-5:]:  # Show last 5 trades
            report.append(f"  {trade['timestamp']} - {trade['side']} {trade['size']} @ ${trade['price']:.2f}")
    else:
        report.append("  No recent trades")
    
    # Recent Signals
    report.append("\n🎯 RECENT SIGNALS:")
    if recent_signals:
        for signal in recent_signals[-5:]:  # Show last 5 signals
            report.append(f"  {signal['timestamp']} - {signal['signal']} (conf: {signal['confidence']:.2f})")
    else:
        report.append("  No recent signals")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)