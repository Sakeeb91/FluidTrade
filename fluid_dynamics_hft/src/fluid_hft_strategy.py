"""
FluidHFTStrategy: A high-frequency trading strategy based on fluid dynamics principles.

This module implements trading strategies that leverage fluid dynamics market models
to generate trading signals and execute trades with optimal timing.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum

from .fluid_market_model import FluidMarketModel


class SignalType(Enum):
    """Types of trading signals."""
    BUY = 1
    SELL = -1
    HOLD = 0


class OrderType(Enum):
    """Types of orders."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    symbol: str
    side: SignalType
    size: float
    price: float
    order_type: OrderType
    trade_id: str = ""
    commission: float = 0.0
    slippage: float = 0.0
    
    def __post_init__(self):
        if not self.trade_id:
            self.trade_id = f"{self.timestamp.strftime('%Y%m%d_%H%M%S')}_{self.symbol}_{self.side.name}"


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    size: float = 0.0
    average_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    last_update: datetime = field(default_factory=datetime.now)
    
    def update_position(self, trade: Trade):
        """Update position with new trade."""
        if trade.side == SignalType.BUY:
            new_size = self.size + trade.size
            if self.size <= 0:  # Opening long or covering short
                self.average_price = trade.price
            else:  # Adding to long
                self.average_price = (self.average_price * self.size + trade.price * trade.size) / new_size
            self.size = new_size
        elif trade.side == SignalType.SELL:
            new_size = self.size - trade.size
            if self.size >= 0:  # Opening short or reducing long
                if self.size > 0:  # Reducing long position
                    self.realized_pnl += (trade.price - self.average_price) * min(trade.size, self.size)
                if new_size < 0:  # Flipping to short
                    self.average_price = trade.price
            else:  # Adding to short
                self.average_price = (self.average_price * abs(self.size) + trade.price * trade.size) / abs(new_size)
            self.size = new_size
        
        self.last_update = trade.timestamp
    
    def update_unrealized_pnl(self, current_price: float):
        """Update unrealized PnL with current market price."""
        if self.size != 0:
            if self.size > 0:  # Long position
                self.unrealized_pnl = (current_price - self.average_price) * self.size
            else:  # Short position
                self.unrealized_pnl = (self.average_price - current_price) * abs(self.size)
        else:
            self.unrealized_pnl = 0.0


class FluidHFTStrategy:
    """
    High-frequency trading strategy based on fluid dynamics market analysis.
    
    This strategy uses flow patterns, pressure gradients, and vorticity to
    identify optimal entry and exit points for trades.
    """
    
    def __init__(self, model: FluidMarketModel, config: Optional[Dict] = None):
        """
        Initialize the FluidHFTStrategy.
        
        Args:
            model: FluidMarketModel instance for market analysis
            config: Strategy configuration parameters
        """
        self.model = model
        self.config = config or {}
        
        # Strategy parameters
        self.position_size_limit = self.config.get('position_size_limit', 100)
        self.max_drawdown = self.config.get('max_drawdown', 0.05)
        self.risk_factor = self.config.get('risk_factor', 0.1)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        self.hold_time_limit = self.config.get('hold_time_limit', 30)  # seconds
        
        # Signal parameters
        self.flow_threshold = self.config.get('flow_threshold', 0.5)
        self.pressure_threshold = self.config.get('pressure_threshold', 0.3)
        self.vorticity_threshold = self.config.get('vorticity_threshold', 0.4)
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.signals: List[Dict] = []
        self.performance_metrics: Dict[str, float] = {}
        
        # Risk management
        self.max_position_size = self.config.get('max_position_size', 1000)
        self.stop_loss_pct = self.config.get('stop_loss_pct', 0.02)
        self.take_profit_pct = self.config.get('take_profit_pct', 0.04)
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def analyze_market_state(self, time_index: int) -> Dict[str, Any]:
        """
        Analyze current market state using fluid dynamics.
        
        Args:
            time_index: Current time index in the model
            
        Returns:
            Dictionary with market analysis
        """
        # Get basic market state
        state = self.model.get_market_state(time_index)
        
        # Get prediction
        prediction = self.model.predict_price_movement(time_index)
        
        # Analyze flow patterns
        flow_analysis = self._analyze_flow_patterns(time_index)
        
        # Analyze pressure dynamics
        pressure_analysis = self._analyze_pressure_dynamics(time_index)
        
        # Analyze vorticity patterns
        vorticity_analysis = self._analyze_vorticity_patterns(time_index)
        
        return {
            'basic_state': state,
            'prediction': prediction,
            'flow_analysis': flow_analysis,
            'pressure_analysis': pressure_analysis,
            'vorticity_analysis': vorticity_analysis,
            'timestamp': time_index
        }
    
    def _analyze_flow_patterns(self, time_index: int) -> Dict[str, Any]:
        """Analyze flow patterns for trading signals."""
        if self.model.flow_field is None:
            return {}
        
        current_flow = self.model.flow_field[time_index, :]
        
        # Flow direction and strength
        flow_direction = np.sign(np.mean(current_flow))
        flow_strength = np.mean(np.abs(current_flow))
        flow_concentration = np.std(current_flow)
        
        # Flow acceleration (compare with previous time steps)
        if time_index > 0:
            prev_flow = self.model.flow_field[time_index - 1, :]
            flow_acceleration = np.mean(current_flow - prev_flow)
        else:
            flow_acceleration = 0.0
        
        # Detect flow reversals
        flow_reversal = False
        if time_index > 2:
            recent_flows = self.model.flow_field[time_index-2:time_index+1, :]
            flow_directions = [np.sign(np.mean(flow)) for flow in recent_flows]
            if len(set(flow_directions)) > 1:
                flow_reversal = True
        
        return {
            'direction': flow_direction,
            'strength': flow_strength,
            'concentration': flow_concentration,
            'acceleration': flow_acceleration,
            'reversal': flow_reversal
        }
    
    def _analyze_pressure_dynamics(self, time_index: int) -> Dict[str, Any]:
        """Analyze pressure dynamics for trading signals."""
        if self.model.pressure_field is None:
            return {}
        
        current_pressure = self.model.pressure_field[time_index, :]
        
        # Pressure statistics
        pressure_level = np.mean(current_pressure)
        pressure_gradient = np.max(np.abs(np.gradient(current_pressure)))
        pressure_variance = np.var(current_pressure)
        
        # Pressure trend
        pressure_trend = 0.0
        if time_index > 4:
            recent_pressures = [np.mean(self.model.pressure_field[i, :]) 
                              for i in range(time_index-4, time_index+1)]
            pressure_trend = np.polyfit(range(5), recent_pressures, 1)[0]
        
        # Pressure extremes
        pressure_peaks = []
        pressure_valleys = []
        if len(current_pressure) > 3:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(current_pressure, height=np.percentile(current_pressure, 70))
            valleys, _ = find_peaks(-current_pressure, height=-np.percentile(current_pressure, 30))
            pressure_peaks = peaks.tolist()
            pressure_valleys = valleys.tolist()
        
        return {
            'level': pressure_level,
            'gradient': pressure_gradient,
            'variance': pressure_variance,
            'trend': pressure_trend,
            'peaks': pressure_peaks,
            'valleys': pressure_valleys
        }
    
    def _analyze_vorticity_patterns(self, time_index: int) -> Dict[str, Any]:
        """Analyze vorticity patterns for trading signals."""
        if self.model.vorticity_field is None:
            return {}
        
        current_vorticity = self.model.vorticity_field[time_index, :]
        
        # Vorticity statistics
        vorticity_magnitude = np.mean(np.abs(current_vorticity))
        vorticity_direction = np.sign(np.mean(current_vorticity))
        vorticity_concentration = np.std(current_vorticity)
        
        # Detect vortex structures
        vortex_strength = np.max(np.abs(current_vorticity))
        vortex_locations = []
        if len(current_vorticity) > 3:
            from scipy.signal import find_peaks
            pos_peaks, _ = find_peaks(current_vorticity, height=np.percentile(current_vorticity, 80))
            neg_peaks, _ = find_peaks(-current_vorticity, height=-np.percentile(current_vorticity, 20))
            vortex_locations = pos_peaks.tolist() + neg_peaks.tolist()
        
        return {
            'magnitude': vorticity_magnitude,
            'direction': vorticity_direction,
            'concentration': vorticity_concentration,
            'strength': vortex_strength,
            'locations': vortex_locations
        }
    
    def generate_signal(self, time_index: int) -> Dict[str, Any]:
        """
        Generate trading signal based on fluid dynamics analysis.
        
        Args:
            time_index: Current time index
            
        Returns:
            Dictionary with signal information
        """
        analysis = self.analyze_market_state(time_index)
        
        # Extract key metrics
        flow_strength = analysis['flow_analysis'].get('strength', 0)
        flow_direction = analysis['flow_analysis'].get('direction', 0)
        flow_acceleration = analysis['flow_analysis'].get('acceleration', 0)
        
        pressure_level = analysis['pressure_analysis'].get('level', 0)
        pressure_gradient = analysis['pressure_analysis'].get('gradient', 0)
        pressure_trend = analysis['pressure_analysis'].get('trend', 0)
        
        vorticity_magnitude = analysis['vorticity_analysis'].get('magnitude', 0)
        vorticity_strength = analysis['vorticity_analysis'].get('strength', 0)
        
        prediction = analysis['prediction']
        
        # Signal generation logic
        signal_strength = 0.0
        signal_direction = SignalType.HOLD
        confidence = 0.0
        
        # Flow-based signals
        if flow_strength > self.flow_threshold:
            if flow_direction > 0 and flow_acceleration > 0:
                signal_strength += 0.3 * flow_strength
                signal_direction = SignalType.BUY
            elif flow_direction < 0 and flow_acceleration < 0:
                signal_strength += 0.3 * flow_strength
                signal_direction = SignalType.SELL
        
        # Pressure-based signals
        if abs(pressure_gradient) > self.pressure_threshold:
            if pressure_trend > 0:
                signal_strength += 0.2 * abs(pressure_gradient)
                if signal_direction == SignalType.HOLD:
                    signal_direction = SignalType.BUY
            elif pressure_trend < 0:
                signal_strength += 0.2 * abs(pressure_gradient)
                if signal_direction == SignalType.HOLD:
                    signal_direction = SignalType.SELL
        
        # Vorticity-based signals (contrarian)
        if vorticity_magnitude > self.vorticity_threshold:
            if vorticity_strength > 0.5:
                signal_strength += 0.1 * vorticity_magnitude
                # Vorticity often indicates reversal
                if signal_direction == SignalType.BUY:
                    signal_direction = SignalType.SELL
                elif signal_direction == SignalType.SELL:
                    signal_direction = SignalType.BUY
        
        # Prediction-based adjustment
        if prediction['confidence'] > 0.5:
            if prediction['price_direction'] > 0:
                if signal_direction == SignalType.SELL:
                    signal_strength *= 0.5  # Reduce conflicting signal
                elif signal_direction == SignalType.BUY:
                    signal_strength *= 1.2  # Enhance agreeing signal
            elif prediction['price_direction'] < 0:
                if signal_direction == SignalType.BUY:
                    signal_strength *= 0.5
                elif signal_direction == SignalType.SELL:
                    signal_strength *= 1.2
        
        # Calculate final confidence
        confidence = min(1.0, signal_strength + 0.3 * prediction['confidence'])
        
        # Apply minimum confidence threshold
        if confidence < self.min_confidence:
            signal_direction = SignalType.HOLD
            signal_strength = 0.0
        
        signal = {
            'timestamp': time_index,
            'signal': signal_direction,
            'strength': signal_strength,
            'confidence': confidence,
            'analysis': analysis,
            'reasoning': self._get_signal_reasoning(analysis, signal_direction, confidence)
        }
        
        self.signals.append(signal)
        return signal
    
    def _get_signal_reasoning(self, analysis: Dict, signal: SignalType, confidence: float) -> str:
        """Generate human-readable reasoning for the signal."""
        reasoning = []
        
        flow_strength = analysis['flow_analysis'].get('strength', 0)
        flow_direction = analysis['flow_analysis'].get('direction', 0)
        pressure_trend = analysis['pressure_analysis'].get('trend', 0)
        vorticity_magnitude = analysis['vorticity_analysis'].get('magnitude', 0)
        
        if signal == SignalType.BUY:
            reasoning.append("BUY signal generated due to:")
            if flow_direction > 0 and flow_strength > self.flow_threshold:
                reasoning.append(f"- Strong positive flow ({flow_strength:.2f})")
            if pressure_trend > 0:
                reasoning.append(f"- Rising pressure trend ({pressure_trend:.2f})")
        elif signal == SignalType.SELL:
            reasoning.append("SELL signal generated due to:")
            if flow_direction < 0 and flow_strength > self.flow_threshold:
                reasoning.append(f"- Strong negative flow ({flow_strength:.2f})")
            if pressure_trend < 0:
                reasoning.append(f"- Declining pressure trend ({pressure_trend:.2f})")
        else:
            reasoning.append("HOLD signal - insufficient confidence or conflicting indicators")
        
        if vorticity_magnitude > self.vorticity_threshold:
            reasoning.append(f"- High vorticity detected ({vorticity_magnitude:.2f}) - potential reversal")
        
        reasoning.append(f"- Overall confidence: {confidence:.2f}")
        
        return "\n".join(reasoning)
    
    def calculate_position_size(self, signal: Dict, current_price: float) -> float:
        """
        Calculate optimal position size based on signal strength and risk management.
        
        Args:
            signal: Trading signal dictionary
            current_price: Current market price
            
        Returns:
            Position size
        """
        if signal['signal'] == SignalType.HOLD:
            return 0.0
        
        # Base position size from signal strength
        base_size = signal['strength'] * self.position_size_limit
        
        # Apply risk factor
        risk_adjusted_size = base_size * self.risk_factor
        
        # Apply confidence scaling
        confidence_adjusted_size = risk_adjusted_size * signal['confidence']
        
        # Apply maximum position limits
        final_size = min(confidence_adjusted_size, self.max_position_size)
        
        return final_size
    
    def execute_trade(self, signal: Dict, current_price: float, symbol: str, 
                     timestamp: datetime) -> Optional[Trade]:
        """
        Execute a trade based on the signal.
        
        Args:
            signal: Trading signal
            current_price: Current market price
            symbol: Trading symbol
            timestamp: Trade timestamp
            
        Returns:
            Trade object if executed, None otherwise
        """
        if signal['signal'] == SignalType.HOLD:
            return None
        
        # Calculate position size
        position_size = self.calculate_position_size(signal, current_price)
        
        if position_size <= 0:
            return None
        
        # Check risk limits
        if not self._check_risk_limits(signal, position_size, current_price, symbol):
            return None
        
        # Create trade
        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=signal['signal'],
            size=position_size,
            price=current_price,
            order_type=OrderType.MARKET,
            commission=0.001 * position_size * current_price,  # 0.1% commission
            slippage=0.0001 * current_price  # 1 basis point slippage
        )
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        
        self.positions[symbol].update_position(trade)
        self.trades.append(trade)
        
        self.logger.info(f"Executed trade: {trade.side.name} {trade.size} {symbol} @ {trade.price}")
        
        return trade
    
    def _check_risk_limits(self, signal: Dict, position_size: float, 
                          current_price: float, symbol: str) -> bool:
        """Check if trade passes risk management limits."""
        # Check position size limits
        if position_size > self.max_position_size:
            return False
        
        # Check current position
        current_position = self.positions.get(symbol, Position(symbol))
        new_position_size = current_position.size
        
        if signal['signal'] == SignalType.BUY:
            new_position_size += position_size
        elif signal['signal'] == SignalType.SELL:
            new_position_size -= position_size
        
        # Check maximum position limit
        if abs(new_position_size) > self.max_position_size:
            return False
        
        # Check drawdown limits
        if self._calculate_current_drawdown() > self.max_drawdown:
            return False
        
        return True
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown."""
        if not self.trades:
            return 0.0
        
        # Calculate running PnL
        running_pnl = 0.0
        max_pnl = 0.0
        max_drawdown = 0.0
        
        for trade in self.trades:
            if trade.side == SignalType.BUY:
                running_pnl -= trade.price * trade.size
            else:
                running_pnl += trade.price * trade.size
            
            running_pnl -= trade.commission
            
            max_pnl = max(max_pnl, running_pnl)
            current_drawdown = (max_pnl - running_pnl) / max(max_pnl, 1.0)
            max_drawdown = max(max_drawdown, current_drawdown)
        
        return max_drawdown
    
    def update_positions(self, current_prices: Dict[str, float]):
        """Update all positions with current market prices."""
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position.update_unrealized_pnl(current_prices[symbol])
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Calculate and return performance metrics."""
        if not self.trades:
            return {'total_trades': 0}
        
        # Calculate basic metrics
        total_trades = len(self.trades)
        buy_trades = sum(1 for trade in self.trades if trade.side == SignalType.BUY)
        sell_trades = sum(1 for trade in self.trades if trade.side == SignalType.SELL)
        
        # Calculate PnL
        total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        total_pnl = total_realized_pnl + total_unrealized_pnl
        
        # Calculate other metrics
        total_commission = sum(trade.commission for trade in self.trades)
        total_volume = sum(trade.size * trade.price for trade in self.trades)
        
        metrics = {
            'total_trades': total_trades,
            'buy_trades': buy_trades,
            'sell_trades': sell_trades,
            'total_pnl': total_pnl,
            'realized_pnl': total_realized_pnl,
            'unrealized_pnl': total_unrealized_pnl,
            'total_commission': total_commission,
            'total_volume': total_volume,
            'avg_trade_size': total_volume / total_trades if total_trades > 0 else 0,
            'current_drawdown': self._calculate_current_drawdown(),
            'win_rate': 0.0  # Would need to track individual trade PnL for this
        }
        
        self.performance_metrics = metrics
        return metrics