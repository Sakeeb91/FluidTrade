"""
FluidMarketSystem: Orchestration system for fluid dynamics-based high-frequency trading.

This module provides the main orchestration system that coordinates market data ingestion,
model updates, signal generation, and trade execution in a unified framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
from datetime import datetime, timedelta
import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum

from .fluid_market_model import FluidMarketModel
from .fluid_hft_strategy import FluidHFTStrategy, SignalType, Trade
from .pattern_identification import FluidPatternRecognizer


class SystemState(Enum):
    """System states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class MarketTick:
    """Represents a market data tick."""
    timestamp: datetime
    symbol: str
    price: float
    volume: float
    bid: float = 0.0
    ask: float = 0.0
    
    def __post_init__(self):
        if self.bid == 0.0:
            self.bid = self.price - 0.01
        if self.ask == 0.0:
            self.ask = self.price + 0.01


@dataclass
class SystemMetrics:
    """System performance metrics."""
    uptime: float = 0.0
    data_points_processed: int = 0
    signals_generated: int = 0
    trades_executed: int = 0
    total_pnl: float = 0.0
    current_drawdown: float = 0.0
    error_count: int = 0
    last_update: datetime = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = datetime.now()


class FluidMarketSystem:
    """
    Main orchestration system for fluid dynamics-based HFT trading.
    
    This system coordinates:
    - Market data ingestion and processing
    - Fluid dynamics model updates
    - Pattern recognition
    - Signal generation
    - Trade execution
    - Risk management
    - Performance monitoring
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the FluidMarketSystem.
        
        Args:
            config: System configuration dictionary
        """
        self.config = config or {}
        
        # Core components
        self.model = FluidMarketModel(config=self.config.get('model', {}))
        self.strategy = FluidHFTStrategy(self.model, config=self.config.get('strategy', {}))
        self.pattern_recognizer = FluidPatternRecognizer(config=self.config.get('patterns', {}))
        
        # System state
        self.state = SystemState.STOPPED
        self.start_time: Optional[datetime] = None
        self.last_update: Optional[datetime] = None
        
        # Data management
        self.market_data: List[MarketTick] = []
        self.data_buffer_size = self.config.get('data_buffer_size', 10000)
        self.update_frequency = self.config.get('update_frequency', 1.0)  # seconds
        
        # Threading
        self.main_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Callbacks
        self.on_signal_callbacks: List[Callable] = []
        self.on_trade_callbacks: List[Callable] = []
        self.on_error_callbacks: List[Callable] = []
        
        # Metrics
        self.metrics = SystemMetrics()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Risk management
        self.risk_limits = self.config.get('risk_limits', {})
        self.emergency_stop_triggered = False
        
    def start(self):
        """Start the trading system."""
        if self.state == SystemState.RUNNING:
            self.logger.warning("System is already running")
            return
        
        self.logger.info("Starting FluidMarketSystem...")
        
        # Initialize components
        self._initialize_components()
        
        # Start main processing thread
        self.stop_event.clear()
        self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self.main_thread.start()
        
        self.state = SystemState.RUNNING
        self.start_time = datetime.now()
        self.logger.info("FluidMarketSystem started successfully")
    
    def stop(self):
        """Stop the trading system."""
        if self.state == SystemState.STOPPED:
            return
        
        self.logger.info("Stopping FluidMarketSystem...")
        
        # Signal threads to stop
        self.stop_event.set()
        
        # Wait for main thread to finish
        if self.main_thread and self.main_thread.is_alive():
            self.main_thread.join(timeout=5.0)
        
        self.state = SystemState.STOPPED
        self.logger.info("FluidMarketSystem stopped")
    
    def pause(self):
        """Pause the trading system."""
        if self.state == SystemState.RUNNING:
            self.state = SystemState.PAUSED
            self.logger.info("FluidMarketSystem paused")
    
    def resume(self):
        """Resume the trading system."""
        if self.state == SystemState.PAUSED:
            self.state = SystemState.RUNNING
            self.logger.info("FluidMarketSystem resumed")
    
    def _initialize_components(self):
        """Initialize all system components."""
        # Load initial data
        if not self.market_data:
            self._load_initial_data()
        
        # Initialize model with data
        if self.market_data:
            self._update_model_data()
        
        # Initialize pattern recognizer
        self.pattern_recognizer.initialize()
        
        self.logger.info("System components initialized")
    
    def _load_initial_data(self):
        """Load initial market data."""
        # Generate synthetic data for demonstration
        self.logger.info("Loading initial market data...")
        
        # Generate synthetic ticks
        n_ticks = 1000
        base_price = 100.0
        start_time = datetime.now() - timedelta(hours=1)
        
        for i in range(n_ticks):
            # Random walk with mean reversion
            price_change = np.random.normal(0, 0.01) - 0.001 * (base_price - 100.0)
            base_price += price_change
            base_price = max(base_price, 50.0)  # Price floor
            
            tick = MarketTick(
                timestamp=start_time + timedelta(seconds=i*3.6),
                symbol="SYNTHETIC",
                price=round(base_price, 2),
                volume=np.random.randint(100, 1000)
            )
            self.market_data.append(tick)
        
        self.logger.info(f"Loaded {len(self.market_data)} market data points")
    
    def _update_model_data(self):
        """Update the market model with current data."""
        if not self.market_data:
            return
        
        # Convert ticks to model format
        df_data = pd.DataFrame([
            {
                'Time': tick.timestamp,
                'Price': tick.price,
                'Size': tick.volume,
                'Type': 1,  # Default type
                'Order_ID': i,
                'Direction': 1 if i % 2 == 0 else -1
            }
            for i, tick in enumerate(self.market_data[-self.data_buffer_size:])
        ])
        
        self.model.data = df_data
        
        # Update fluid dynamics fields
        self.model.compute_flow_field()
        self.model.compute_pressure_field()
        self.model.compute_vorticity_field()
    
    def _main_loop(self):
        """Main processing loop."""
        self.logger.info("Starting main processing loop")
        
        while not self.stop_event.is_set():
            try:
                if self.state == SystemState.RUNNING:
                    self._process_iteration()
                elif self.state == SystemState.PAUSED:
                    time.sleep(0.1)
                    continue
                
                # Sleep until next update
                time.sleep(self.update_frequency)
                
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                self.metrics.error_count += 1
                self._handle_error(e)
                
                # Pause system on repeated errors
                if self.metrics.error_count > 5:
                    self.state = SystemState.ERROR
                    self.logger.error("Too many errors, stopping system")
                    break
        
        self.logger.info("Main processing loop stopped")
    
    def _process_iteration(self):
        """Process one iteration of the system."""
        current_time = datetime.now()
        
        # Update metrics
        if self.start_time:
            self.metrics.uptime = (current_time - self.start_time).total_seconds()
        self.metrics.last_update = current_time
        
        # Generate new market data (in real system, this would come from feed)
        self._generate_new_data()
        
        # Update model
        self._update_model_data()
        
        # Update pattern recognition
        self._update_patterns()
        
        # Generate trading signals
        signals = self._generate_signals()
        
        # Execute trades
        self._execute_trades(signals)
        
        # Update risk management
        self._update_risk_management()
        
        # Update performance metrics
        self._update_metrics()
        
        self.last_update = current_time
    
    def _generate_new_data(self):
        """Generate new market data tick."""
        if not self.market_data:
            return
        
        # Get last tick
        last_tick = self.market_data[-1]
        
        # Generate new tick with realistic price movement
        price_change = np.random.normal(0, 0.005)
        new_price = max(last_tick.price + price_change, 0.01)
        
        new_tick = MarketTick(
            timestamp=datetime.now(),
            symbol=last_tick.symbol,
            price=round(new_price, 2),
            volume=np.random.randint(100, 1000)
        )
        
        self.market_data.append(new_tick)
        
        # Maintain buffer size
        if len(self.market_data) > self.data_buffer_size:
            self.market_data = self.market_data[-self.data_buffer_size:]
        
        self.metrics.data_points_processed += 1
    
    def _update_patterns(self):
        """Update pattern recognition."""
        if (self.model.flow_field is not None and 
            self.model.pressure_field is not None and 
            self.model.vorticity_field is not None):
            
            # Detect patterns
            patterns = self.pattern_recognizer.detect_patterns(
                self.model.flow_field,
                self.model.pressure_field,
                self.model.vorticity_field
            )
            
            # Update model patterns
            self.model.patterns = patterns
    
    def _generate_signals(self) -> List[Dict]:
        """Generate trading signals."""
        signals = []
        
        if self.model.flow_field is not None:
            # Get current time index (last available)
            current_index = min(self.model.time_window - 1, len(self.market_data) - 1)
            
            if current_index > 0:
                # Generate signal
                signal = self.strategy.generate_signal(current_index)
                signals.append(signal)
                
                self.metrics.signals_generated += 1
                
                # Call signal callbacks
                for callback in self.on_signal_callbacks:
                    try:
                        callback(signal)
                    except Exception as e:
                        self.logger.error(f"Error in signal callback: {e}")
        
        return signals
    
    def _execute_trades(self, signals: List[Dict]):
        """Execute trades based on signals."""
        if not signals or not self.market_data:
            return
        
        current_price = self.market_data[-1].price
        current_time = self.market_data[-1].timestamp
        symbol = self.market_data[-1].symbol
        
        for signal in signals:
            if signal['signal'] != SignalType.HOLD:
                # Check if trading is allowed
                if self._is_trading_allowed(signal):
                    try:
                        trade = self.strategy.execute_trade(
                            signal, current_price, symbol, current_time
                        )
                        
                        if trade:
                            self.metrics.trades_executed += 1
                            self.logger.info(f"Executed trade: {trade}")
                            
                            # Call trade callbacks
                            for callback in self.on_trade_callbacks:
                                try:
                                    callback(trade)
                                except Exception as e:
                                    self.logger.error(f"Error in trade callback: {e}")
                    
                    except Exception as e:
                        self.logger.error(f"Error executing trade: {e}")
                        self.metrics.error_count += 1
    
    def _is_trading_allowed(self, signal: Dict) -> bool:
        """Check if trading is allowed based on risk limits."""
        if self.emergency_stop_triggered:
            return False
        
        # Check drawdown limits
        if self.metrics.current_drawdown > self.risk_limits.get('max_drawdown', 0.1):
            self.logger.warning("Trading disabled due to drawdown limit")
            return False
        
        # Check signal confidence
        if signal['confidence'] < self.risk_limits.get('min_confidence', 0.5):
            return False
        
        # Check position limits
        # This would include checks for maximum position size, concentration, etc.
        
        return True
    
    def _update_risk_management(self):
        """Update risk management checks."""
        # Update position values
        if self.market_data:
            current_prices = {self.market_data[-1].symbol: self.market_data[-1].price}
            self.strategy.update_positions(current_prices)
        
        # Check for emergency stop conditions
        performance = self.strategy.get_performance_metrics()
        
        if performance.get('current_drawdown', 0) > self.risk_limits.get('emergency_stop_drawdown', 0.2):
            self.emergency_stop_triggered = True
            self.logger.error("Emergency stop triggered due to excessive drawdown")
            self.state = SystemState.ERROR
    
    def _update_metrics(self):
        """Update system metrics."""
        performance = self.strategy.get_performance_metrics()
        
        self.metrics.total_pnl = performance.get('total_pnl', 0.0)
        self.metrics.current_drawdown = performance.get('current_drawdown', 0.0)
    
    def _handle_error(self, error: Exception):
        """Handle system errors."""
        self.logger.error(f"System error: {error}")
        
        # Call error callbacks
        for callback in self.on_error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"Error in error callback: {e}")
    
    def add_signal_callback(self, callback: Callable):
        """Add callback for signal generation events."""
        self.on_signal_callbacks.append(callback)
    
    def add_trade_callback(self, callback: Callable):
        """Add callback for trade execution events."""
        self.on_trade_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable):
        """Add callback for error events."""
        self.on_error_callbacks.append(callback)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'state': self.state.value,
            'uptime': self.metrics.uptime,
            'data_points': self.metrics.data_points_processed,
            'signals_generated': self.metrics.signals_generated,
            'trades_executed': self.metrics.trades_executed,
            'total_pnl': self.metrics.total_pnl,
            'current_drawdown': self.metrics.current_drawdown,
            'error_count': self.metrics.error_count,
            'last_update': self.metrics.last_update,
            'emergency_stop': self.emergency_stop_triggered,
            'positions': len(self.strategy.positions),
            'data_buffer_size': len(self.market_data)
        }
    
    def get_current_positions(self) -> Dict[str, Dict]:
        """Get current trading positions."""
        return {
            symbol: {
                'size': pos.size,
                'average_price': pos.average_price,
                'unrealized_pnl': pos.unrealized_pnl,
                'realized_pnl': pos.realized_pnl
            }
            for symbol, pos in self.strategy.positions.items()
            if pos.size != 0
        }
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict]:
        """Get recent trades."""
        recent_trades = self.strategy.trades[-limit:] if self.strategy.trades else []
        return [
            {
                'timestamp': trade.timestamp,
                'symbol': trade.symbol,
                'side': trade.side.name,
                'size': trade.size,
                'price': trade.price,
                'trade_id': trade.trade_id
            }
            for trade in recent_trades
        ]
    
    def get_recent_signals(self, limit: int = 10) -> List[Dict]:
        """Get recent signals."""
        recent_signals = self.strategy.signals[-limit:] if self.strategy.signals else []
        return [
            {
                'timestamp': signal['timestamp'],
                'signal': signal['signal'].name if hasattr(signal['signal'], 'name') else str(signal['signal']),
                'strength': signal['strength'],
                'confidence': signal['confidence'],
                'reasoning': signal.get('reasoning', '')
            }
            for signal in recent_signals
        ]
    
    def run_backtest(self, start_date: str, end_date: str, 
                    initial_capital: float = 100000.0) -> Dict[str, Any]:
        """
        Run a backtest over historical data.
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital for backtest
            
        Returns:
            Backtest results
        """
        self.logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # For now, simulate backtest with synthetic data
        # In real implementation, this would load historical data
        
        # Generate synthetic historical data
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        # Create synthetic price series
        trading_days = pd.date_range(start=start_dt, end=end_dt, freq='B')  # Business days
        
        backtest_results = {
            'start_date': start_date,
            'end_date': end_date,
            'initial_capital': initial_capital,
            'final_capital': initial_capital,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': 0,
            'win_rate': 0.0,
            'daily_returns': []
        }
        
        # Simulate trading over the period
        capital = initial_capital
        daily_returns = []
        
        for day in trading_days:
            # Generate synthetic daily return
            daily_return = np.random.normal(0.0005, 0.02)  # 0.05% daily return with 2% volatility
            daily_returns.append(daily_return)
            capital *= (1 + daily_return)
        
        backtest_results['final_capital'] = capital
        backtest_results['total_return'] = (capital - initial_capital) / initial_capital
        backtest_results['daily_returns'] = daily_returns
        
        # Calculate additional metrics
        if daily_returns:
            returns_array = np.array(daily_returns)
            backtest_results['sharpe_ratio'] = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252)
            
            # Calculate max drawdown
            cumulative_returns = np.cumprod(1 + returns_array)
            peak = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - peak) / peak
            backtest_results['max_drawdown'] = abs(np.min(drawdown))
        
        self.logger.info(f"Backtest completed. Total return: {backtest_results['total_return']:.2%}")
        
        return backtest_results