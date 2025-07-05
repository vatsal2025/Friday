import logging
import numpy as np
import pandas as pd
import random
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

from src.services.data.market_data import MarketData
from src.services.model.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


class TraderProfile:
    """Class representing a simulated trader's profile and behavior patterns."""
    
    def __init__(self, 
                 name: str,
                 risk_tolerance: float,  # 0.0 to 1.0 (conservative to aggressive)
                 holding_period: Tuple[int, int],  # (min_days, max_days)
                 profit_target: float,  # target profit percentage
                 stop_loss: float,  # stop loss percentage
                 preferred_sectors: List[str] = None,
                 preferred_indicators: List[str] = None,
                 trading_frequency: float = 0.5,  # 0.0 to 1.0 (infrequent to frequent)
                 position_sizing: float = 0.1,  # percentage of portfolio per trade
                 expertise_level: float = 0.7  # 0.0 to 1.0 (novice to expert)
                ):
        """Initialize a trader profile.
        
        Args:
            name: Name of the trader profile.
            risk_tolerance: Risk tolerance level (0.0 to 1.0).
            holding_period: Tuple of (min_days, max_days) for holding positions.
            profit_target: Target profit percentage for taking profits.
            stop_loss: Stop loss percentage for cutting losses.
            preferred_sectors: List of preferred market sectors.
            preferred_indicators: List of preferred technical indicators.
            trading_frequency: How frequently the trader makes trades (0.0 to 1.0).
            position_sizing: Percentage of portfolio allocated per trade.
            expertise_level: Expertise level of the trader (0.0 to 1.0).
        """
        self.name = name
        self.risk_tolerance = max(0.0, min(1.0, risk_tolerance))
        self.holding_period = holding_period
        self.profit_target = profit_target
        self.stop_loss = stop_loss
        self.preferred_sectors = preferred_sectors or []
        self.preferred_indicators = preferred_indicators or []
        self.trading_frequency = max(0.0, min(1.0, trading_frequency))
        self.position_sizing = max(0.01, min(1.0, position_sizing))
        self.expertise_level = max(0.0, min(1.0, expertise_level))
        
        # Initialize trading history
        self.trading_history = []
        
        logger.info("Created trader profile: %s", name)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trader profile to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the trader profile.
        """
        return {
            "name": self.name,
            "risk_tolerance": self.risk_tolerance,
            "holding_period": self.holding_period,
            "profit_target": self.profit_target,
            "stop_loss": self.stop_loss,
            "preferred_sectors": self.preferred_sectors,
            "preferred_indicators": self.preferred_indicators,
            "trading_frequency": self.trading_frequency,
            "position_sizing": self.position_sizing,
            "expertise_level": self.expertise_level
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraderProfile':
        """Create a trader profile from a dictionary.
        
        Args:
            data: Dictionary containing trader profile data.
            
        Returns:
            TraderProfile: A new trader profile instance.
        """
        return cls(
            name=data["name"],
            risk_tolerance=data["risk_tolerance"],
            holding_period=data["holding_period"],
            profit_target=data["profit_target"],
            stop_loss=data["stop_loss"],
            preferred_sectors=data.get("preferred_sectors", []),
            preferred_indicators=data.get("preferred_indicators", []),
            trading_frequency=data.get("trading_frequency", 0.5),
            position_sizing=data.get("position_sizing", 0.1),
            expertise_level=data.get("expertise_level", 0.7)
        )


class TradeDecision:
    """Class representing a trade decision made by a simulated trader."""
    
    def __init__(self,
                 trader_name: str,
                 symbol: str,
                 action: str,  # 'buy' or 'sell'
                 timestamp: datetime,
                 price: float,
                 quantity: int,
                 confidence: float,
                 reasoning: str = None,
                 indicators_used: List[str] = None,
                 expected_holding_period: int = None,
                 target_price: float = None,
                 stop_loss_price: float = None):
        """Initialize a trade decision.
        
        Args:
            trader_name: Name of the trader making the decision.
            symbol: Trading symbol (e.g., stock ticker).
            action: Trade action ('buy' or 'sell').
            timestamp: Timestamp of the decision.
            price: Price at which the trade is executed.
            quantity: Number of shares/units to trade.
            confidence: Confidence level in the decision (0.0 to 1.0).
            reasoning: Reasoning behind the decision.
            indicators_used: Technical indicators used for the decision.
            expected_holding_period: Expected holding period in days.
            target_price: Target price for taking profits.
            stop_loss_price: Stop loss price for cutting losses.
        """
        self.trader_name = trader_name
        self.symbol = symbol
        self.action = action.lower()
        self.timestamp = timestamp
        self.price = price
        self.quantity = quantity
        self.confidence = confidence
        self.reasoning = reasoning
        self.indicators_used = indicators_used or []
        self.expected_holding_period = expected_holding_period
        self.target_price = target_price
        self.stop_loss_price = stop_loss_price
        
        # Trade outcome (to be filled later)
        self.exit_timestamp = None
        self.exit_price = None
        self.profit_loss = None
        self.profit_loss_pct = None
        self.holding_period = None
        self.exit_reason = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade decision to dictionary.
        
        Returns:
            Dict[str, Any]: Dictionary representation of the trade decision.
        """
        return {
            "trader_name": self.trader_name,
            "symbol": self.symbol,
            "action": self.action,
            "timestamp": self.timestamp.isoformat(),
            "price": self.price,
            "quantity": self.quantity,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "indicators_used": self.indicators_used,
            "expected_holding_period": self.expected_holding_period,
            "target_price": self.target_price,
            "stop_loss_price": self.stop_loss_price,
            "exit_timestamp": self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            "exit_price": self.exit_price,
            "profit_loss": self.profit_loss,
            "profit_loss_pct": self.profit_loss_pct,
            "holding_period": self.holding_period,
            "exit_reason": self.exit_reason
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeDecision':
        """Create a trade decision from a dictionary.
        
        Args:
            data: Dictionary containing trade decision data.
            
        Returns:
            TradeDecision: A new trade decision instance.
        """
        decision = cls(
            trader_name=data["trader_name"],
            symbol=data["symbol"],
            action=data["action"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            price=data["price"],
            quantity=data["quantity"],
            confidence=data["confidence"],
            reasoning=data.get("reasoning"),
            indicators_used=data.get("indicators_used"),
            expected_holding_period=data.get("expected_holding_period"),
            target_price=data.get("target_price"),
            stop_loss_price=data.get("stop_loss_price")
        )
        
        # Set outcome fields if available
        if data.get("exit_timestamp"):
            decision.exit_timestamp = datetime.fromisoformat(data["exit_timestamp"])
        decision.exit_price = data.get("exit_price")
        decision.profit_loss = data.get("profit_loss")
        decision.profit_loss_pct = data.get("profit_loss_pct")
        decision.holding_period = data.get("holding_period")
        decision.exit_reason = data.get("exit_reason")
        
        return decision


class ExperiencedTraderSimulation:
    """Class for simulating experienced traders' behavior and decisions."""
    
    def __init__(self, market_data: MarketData, model_registry: ModelRegistry = None):
        """Initialize a trader simulation.
        
        Args:
            market_data: Market data service for accessing historical data.
            model_registry: Optional model registry for accessing prediction models.
        """
        self.market_data = market_data
        self.model_registry = model_registry
        self.trader_profiles = {}
        self.simulation_results = {}
        logger.info("Initialized ExperiencedTraderSimulation")
    
    def add_trader_profile(self, profile: TraderProfile) -> None:
        """Add a trader profile to the simulation.
        
        Args:
            profile: Trader profile to add.
            
        Raises:
            ValueError: If a profile with the same name already exists.
        """
        if profile.name in self.trader_profiles:
            raise ValueError(f"Trader profile '{profile.name}' already exists")
        
        self.trader_profiles[profile.name] = profile
        logger.info("Added trader profile: %s", profile.name)
    
    def create_default_profiles(self) -> None:
        """Create a set of default trader profiles with different characteristics."""
        # Conservative trader
        self.add_trader_profile(TraderProfile(
            name="Conservative Investor",
            risk_tolerance=0.2,
            holding_period=(30, 180),  # 1-6 months
            profit_target=0.15,  # 15%
            stop_loss=0.05,  # 5%
            preferred_sectors=["Utilities", "Consumer Staples", "Healthcare"],
            preferred_indicators=["Moving Average", "RSI", "Dividend Yield"],
            trading_frequency=0.2,
            position_sizing=0.05,
            expertise_level=0.8
        ))
        
        # Moderate trader
        self.add_trader_profile(TraderProfile(
            name="Balanced Trader",
            risk_tolerance=0.5,
            holding_period=(14, 90),  # 2 weeks to 3 months
            profit_target=0.25,  # 25%
            stop_loss=0.1,  # 10%
            preferred_sectors=["Technology", "Financials", "Industrials"],
            preferred_indicators=["MACD", "Bollinger Bands", "Volume"],
            trading_frequency=0.5,
            position_sizing=0.1,
            expertise_level=0.7
        ))
        
        # Aggressive trader
        self.add_trader_profile(TraderProfile(
            name="Aggressive Trader",
            risk_tolerance=0.8,
            holding_period=(1, 30),  # 1 day to 1 month
            profit_target=0.4,  # 40%
            stop_loss=0.15,  # 15%
            preferred_sectors=["Technology", "Biotech", "Crypto"],
            preferred_indicators=["RSI", "Momentum", "Stochastic"],
            trading_frequency=0.8,
            position_sizing=0.2,
            expertise_level=0.9
        ))
        
        # Technical analyst
        self.add_trader_profile(TraderProfile(
            name="Technical Analyst",
            risk_tolerance=0.6,
            holding_period=(5, 60),  # 5 days to 2 months
            profit_target=0.3,  # 30%
            stop_loss=0.1,  # 10%
            preferred_sectors=[],  # No sector preference, purely technical
            preferred_indicators=["Moving Average", "RSI", "MACD", "Fibonacci", "Chart Patterns"],
            trading_frequency=0.7,
            position_sizing=0.15,
            expertise_level=0.85
        ))
        
        # Value investor
        self.add_trader_profile(TraderProfile(
            name="Value Investor",
            risk_tolerance=0.4,
            holding_period=(90, 365),  # 3 months to 1 year
            profit_target=0.3,  # 30%
            stop_loss=0.15,  # 15%
            preferred_sectors=["Financials", "Energy", "Consumer Discretionary"],
            preferred_indicators=["P/E Ratio", "P/B Ratio", "Dividend Yield", "FCF"],
            trading_frequency=0.3,
            position_sizing=0.1,
            expertise_level=0.75
        ))
        
        logger.info("Created default trader profiles")
    
    def simulate_trader_decisions(self,
                                 trader_name: str,
                                 symbols: List[str],
                                 start_date: datetime,
                                 end_date: datetime,
                                 initial_capital: float = 100000.0,
                                 use_model_signals: bool = False,
                                 model_id: str = None) -> Dict[str, Any]:
        """Simulate trading decisions for a specific trader over a time period.
        
        Args:
            trader_name: Name of the trader profile to use.
            symbols: List of symbols to trade.
            start_date: Start date of the simulation.
            end_date: End date of the simulation.
            initial_capital: Initial capital for the simulation.
            use_model_signals: Whether to incorporate model signals into decisions.
            model_id: ID of the model to use for signals (if use_model_signals is True).
            
        Returns:
            Dict[str, Any]: Simulation results.
            
        Raises:
            ValueError: If the trader profile doesn't exist.
        """
        if trader_name not in self.trader_profiles:
            raise ValueError(f"Trader profile '{trader_name}' does not exist")
        
        profile = self.trader_profiles[trader_name]
        
        # Initialize simulation state
        current_date = start_date
        available_capital = initial_capital
        portfolio = {}
        trade_history = []
        daily_portfolio_value = []
        
        # Load model if using model signals
        model = None
        if use_model_signals and self.model_registry and model_id:
            model = self.model_registry.load_model(model_id)
            if model is None:
                logger.warning("Model %s not found, proceeding without model signals", model_id)
                use_model_signals = False
        
        # Simulation loop
        while current_date <= end_date:
            # Skip weekends
            if current_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                current_date += timedelta(days=1)
                continue
            
            # Calculate portfolio value for this day
            portfolio_value = available_capital
            for symbol, position in portfolio.items():
                try:
                    current_price = self._get_price(symbol, current_date)
                    position_value = position["quantity"] * current_price
                    portfolio_value += position_value
                except Exception as e:
                    logger.warning("Error getting price for %s on %s: %s", 
                                 symbol, current_date.strftime("%Y-%m-%d"), e)
            
            daily_portfolio_value.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "value": portfolio_value
            })
            
            # Check for exit conditions on existing positions
            for symbol in list(portfolio.keys()):
                try:
                    current_price = self._get_price(symbol, current_date)
                    position = portfolio[symbol]
                    entry_price = position["price"]
                    entry_date = position["date"]
                    holding_days = (current_date - entry_date).days
                    
                    # Calculate profit/loss percentage
                    pl_pct = (current_price - entry_price) / entry_price
                    
                    exit_reason = None
                    
                    # Check stop loss
                    if pl_pct <= -profile.stop_loss:
                        exit_reason = "stop_loss"
                    
                    # Check profit target
                    elif pl_pct >= profile.profit_target:
                        exit_reason = "profit_target"
                    
                    # Check holding period
                    elif holding_days >= profile.holding_period[1]:
                        exit_reason = "max_holding_period"
                    
                    # Execute exit if needed
                    if exit_reason:
                        # Create exit trade decision
                        exit_decision = TradeDecision(
                            trader_name=trader_name,
                            symbol=symbol,
                            action="sell",
                            timestamp=current_date,
                            price=current_price,
                            quantity=position["quantity"],
                            confidence=self._calculate_exit_confidence(profile, exit_reason, pl_pct),
                            reasoning=f"Exit due to {exit_reason}",
                            indicators_used=position.get("indicators_used", [])
                        )
                        
                        # Update the entry decision with exit information
                        entry_decision = position["decision"]
                        entry_decision.exit_timestamp = current_date
                        entry_decision.exit_price = current_price
                        entry_decision.profit_loss = (current_price - entry_price) * position["quantity"]
                        entry_decision.profit_loss_pct = pl_pct
                        entry_decision.holding_period = holding_days
                        entry_decision.exit_reason = exit_reason
                        
                        # Add capital back to available capital
                        available_capital += position["quantity"] * current_price
                        
                        # Remove position from portfolio
                        del portfolio[symbol]
                        
                        # Add exit decision to trade history
                        trade_history.append(exit_decision)
                        
                        logger.info("%s: Exited %s position at %.2f (%.2f%%), reason: %s", 
                                   trader_name, symbol, current_price, pl_pct * 100, exit_reason)
                except Exception as e:
                    logger.warning("Error processing exit for %s on %s: %s", 
                                 symbol, current_date.strftime("%Y-%m-%d"), e)
            
            # Decide whether to make new trades today based on trading frequency
            if random.random() <= profile.trading_frequency:
                # Determine how many symbols to consider
                num_symbols_to_consider = max(1, int(len(symbols) * 0.3))
                symbols_to_consider = random.sample(symbols, min(num_symbols_to_consider, len(symbols)))
                
                for symbol in symbols_to_consider:
                    # Skip if already in portfolio
                    if symbol in portfolio:
                        continue
                    
                    try:
                        # Get market data for decision making
                        current_price = self._get_price(symbol, current_date)
                        
                        # Determine whether to buy based on trader's strategy
                        buy_signal, confidence, reasoning, indicators = self._generate_buy_signal(
                            profile, symbol, current_date, use_model_signals, model
                        )
                        
                        if buy_signal and confidence > 0.5:  # Only buy if confidence is high enough
                            # Calculate position size
                            position_value = available_capital * profile.position_sizing
                            quantity = int(position_value / current_price)
                            
                            if quantity > 0 and position_value <= available_capital:
                                # Create entry trade decision
                                entry_decision = TradeDecision(
                                    trader_name=trader_name,
                                    symbol=symbol,
                                    action="buy",
                                    timestamp=current_date,
                                    price=current_price,
                                    quantity=quantity,
                                    confidence=confidence,
                                    reasoning=reasoning,
                                    indicators_used=indicators,
                                    expected_holding_period=random.randint(
                                        profile.holding_period[0], profile.holding_period[1]
                                    ),
                                    target_price=current_price * (1 + profile.profit_target),
                                    stop_loss_price=current_price * (1 - profile.stop_loss)
                                )
                                
                                # Add position to portfolio
                                portfolio[symbol] = {
                                    "quantity": quantity,
                                    "price": current_price,
                                    "date": current_date,
                                    "decision": entry_decision,
                                    "indicators_used": indicators
                                }
                                
                                # Subtract from available capital
                                available_capital -= quantity * current_price
                                
                                # Add entry decision to trade history
                                trade_history.append(entry_decision)
                                
                                logger.info("%s: Entered %s position at %.2f, quantity: %d, confidence: %.2f", 
                                           trader_name, symbol, current_price, quantity, confidence)
                    except Exception as e:
                        logger.warning("Error processing entry for %s on %s: %s", 
                                     symbol, current_date.strftime("%Y-%m-%d"), e)
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Close any remaining positions at the end of simulation
        for symbol in list(portfolio.keys()):
            try:
                current_price = self._get_price(symbol, end_date)
                position = portfolio[symbol]
                entry_price = position["price"]
                entry_date = position["date"]
                holding_days = (end_date - entry_date).days
                
                # Calculate profit/loss percentage
                pl_pct = (current_price - entry_price) / entry_price
                
                # Create exit trade decision
                exit_decision = TradeDecision(
                    trader_name=trader_name,
                    symbol=symbol,
                    action="sell",
                    timestamp=end_date,
                    price=current_price,
                    quantity=position["quantity"],
                    confidence=0.5,  # Neutral confidence for end-of-simulation exit
                    reasoning="End of simulation",
                    indicators_used=position.get("indicators_used", [])
                )
                
                # Update the entry decision with exit information
                entry_decision = position["decision"]
                entry_decision.exit_timestamp = end_date
                entry_decision.exit_price = current_price
                entry_decision.profit_loss = (current_price - entry_price) * position["quantity"]
                entry_decision.profit_loss_pct = pl_pct
                entry_decision.holding_period = holding_days
                entry_decision.exit_reason = "end_of_simulation"
                
                # Add capital back to available capital
                available_capital += position["quantity"] * current_price
                
                # Add exit decision to trade history
                trade_history.append(exit_decision)
                
                logger.info("%s: Closed %s position at end of simulation at %.2f (%.2f%%)", 
                           trader_name, symbol, current_price, pl_pct * 100)
            except Exception as e:
                logger.warning("Error closing position for %s at end of simulation: %s", symbol, e)
        
        # Calculate performance metrics
        final_portfolio_value = daily_portfolio_value[-1]["value"] if daily_portfolio_value else initial_capital
        total_return = (final_portfolio_value - initial_capital) / initial_capital
        
        # Calculate win rate
        profitable_trades = sum(1 for trade in trade_history 
                              if trade.action == "buy" and trade.profit_loss_pct is not None 
                              and trade.profit_loss_pct > 0)
        total_closed_trades = sum(1 for trade in trade_history 
                                if trade.action == "buy" and trade.profit_loss_pct is not None)
        win_rate = profitable_trades / total_closed_trades if total_closed_trades > 0 else 0
        
        # Calculate average profit/loss
        avg_profit_loss = sum(trade.profit_loss_pct for trade in trade_history 
                            if trade.action == "buy" and trade.profit_loss_pct is not None) \
                        / total_closed_trades if total_closed_trades > 0 else 0
        
        # Store simulation results
        results = {
            "trader_name": trader_name,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "initial_capital": initial_capital,
            "final_portfolio_value": final_portfolio_value,
            "total_return": total_return,
            "total_trades": total_closed_trades,
            "win_rate": win_rate,
            "avg_profit_loss": avg_profit_loss,
            "trade_history": [trade.to_dict() for trade in trade_history],
            "daily_portfolio_value": daily_portfolio_value
        }
        
        self.simulation_results[f"{trader_name}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"] = results
        
        logger.info("%s simulation completed: %.2f%% return, %.2f%% win rate", 
                   trader_name, total_return * 100, win_rate * 100)
        
        return results
    
    def _get_price(self, symbol: str, date: datetime) -> float:
        """Get the price for a symbol on a specific date.
        
        Args:
            symbol: Trading symbol.
            date: Date to get price for.
            
        Returns:
            float: Price of the symbol on the date.
            
        Raises:
            ValueError: If price data is not available.
        """
        # This is a simplified implementation
        # In a real system, this would use the market_data service to get actual prices
        try:
            # Try to get closing price from market data service
            price_data = self.market_data.get_historical_data(
                symbol=symbol,
                start_date=date,
                end_date=date + timedelta(days=1),
                interval="1d"
            )
            
            if price_data is not None and not price_data.empty:
                return price_data.iloc[0]["close"]
            
            # Fallback to a random price for simulation purposes
            # In a real implementation, this would raise an error
            logger.warning("No price data for %s on %s, using random price", 
                         symbol, date.strftime("%Y-%m-%d"))
            return 100.0 * (1 + 0.1 * np.random.randn())
        
        except Exception as e:
            logger.error("Error getting price for %s on %s: %s", 
                       symbol, date.strftime("%Y-%m-%d"), e)
            # Fallback to a random price for simulation purposes
            return 100.0 * (1 + 0.1 * np.random.randn())
    
    def _generate_buy_signal(self, 
                            profile: TraderProfile, 
                            symbol: str, 
                            date: datetime,
                            use_model_signals: bool = False,
                            model: Any = None) -> Tuple[bool, float, str, List[str]]:
        """Generate a buy signal based on trader profile and market data.
        
        Args:
            profile: Trader profile.
            symbol: Trading symbol.
            date: Current date.
            use_model_signals: Whether to incorporate model signals.
            model: Model to use for signals.
            
        Returns:
            Tuple[bool, float, str, List[str]]: 
                - Buy signal (True/False)
                - Confidence level (0.0 to 1.0)
                - Reasoning behind the decision
                - List of indicators used
        """
        # This is a simplified implementation that generates synthetic signals
        # In a real system, this would use actual technical indicators and fundamental data
        
        # Initialize variables
        buy_signal = False
        confidence = 0.0
        reasoning = ""
        indicators_used = []
        
        try:
            # Get historical data for analysis
            lookback_days = 30  # Adjust based on trader profile
            start_date = date - timedelta(days=lookback_days)
            
            historical_data = self.market_data.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=date,
                interval="1d"
            )
            
            if historical_data is None or historical_data.empty:
                # Not enough data for analysis
                return False, 0.0, "Insufficient data", []
            
            # Select indicators based on trader preferences
            available_indicators = ["Moving Average", "RSI", "MACD", "Bollinger Bands", 
                                  "Volume", "Momentum", "Stochastic"]
            
            # Use preferred indicators if available, otherwise use random ones
            if profile.preferred_indicators:
                selected_indicators = profile.preferred_indicators
            else:
                num_indicators = random.randint(2, 4)
                selected_indicators = random.sample(available_indicators, num_indicators)
            
            indicators_used = selected_indicators
            
            # Generate synthetic indicator signals
            indicator_signals = {}
            indicator_confidences = {}
            
            for indicator in selected_indicators:
                # Simulate indicator calculation and signal generation
                # In a real system, this would use actual technical analysis libraries
                if indicator == "Moving Average":
                    # Simple moving average crossover
                    if len(historical_data) >= 20:
                        short_ma = historical_data["close"].rolling(window=5).mean().iloc[-1]
                        long_ma = historical_data["close"].rolling(window=20).mean().iloc[-1]
                        signal = short_ma > long_ma
                        strength = abs(short_ma - long_ma) / long_ma
                        indicator_signals[indicator] = signal
                        indicator_confidences[indicator] = min(1.0, strength * 10)
                        reasoning += f"Moving Average: {'Bullish' if signal else 'Bearish'} (short MA {short_ma:.2f} vs long MA {long_ma:.2f}). "
                
                elif indicator == "RSI":
                    # Relative Strength Index
                    if len(historical_data) >= 14:
                        # Simplified RSI calculation
                        delta = historical_data["close"].diff()
                        gain = delta.where(delta > 0, 0).rolling(window=14).mean().iloc[-1]
                        loss = -delta.where(delta < 0, 0).rolling(window=14).mean().iloc[-1]
                        rs = gain / loss if loss != 0 else 0
                        rsi = 100 - (100 / (1 + rs))
                        
                        # RSI below 30 is oversold (buy), above 70 is overbought (sell)
                        signal = rsi < 30 + (40 * profile.risk_tolerance)  # Adjust threshold based on risk tolerance
                        strength = (70 - rsi) / 40 if rsi < 70 else 0
                        indicator_signals[indicator] = signal
                        indicator_confidences[indicator] = min(1.0, max(0.0, strength))
                        reasoning += f"RSI: {'Bullish' if signal else 'Bearish'} (RSI = {rsi:.2f}). "
                
                elif indicator == "MACD":
                    # Moving Average Convergence Divergence
                    if len(historical_data) >= 26:
                        ema12 = historical_data["close"].ewm(span=12).mean()
                        ema26 = historical_data["close"].ewm(span=26).mean()
                        macd = ema12 - ema26
                        signal_line = macd.ewm(span=9).mean()
                        
                        # MACD crossing above signal line is bullish
                        current_macd = macd.iloc[-1]
                        current_signal = signal_line.iloc[-1]
                        prev_macd = macd.iloc[-2]
                        prev_signal = signal_line.iloc[-2]
                        
                        signal = (current_macd > current_signal) and (prev_macd <= prev_signal)
                        strength = abs(current_macd - current_signal) / abs(current_signal) if current_signal != 0 else 0.5
                        indicator_signals[indicator] = signal
                        indicator_confidences[indicator] = min(1.0, strength * 5)
                        reasoning += f"MACD: {'Bullish' if signal else 'Bearish'} (MACD = {current_macd:.4f}, Signal = {current_signal:.4f}). "
                
                elif indicator == "Bollinger Bands":
                    # Bollinger Bands
                    if len(historical_data) >= 20:
                        sma = historical_data["close"].rolling(window=20).mean()
                        std = historical_data["close"].rolling(window=20).std()
                        upper_band = sma + (2 * std)
                        lower_band = sma - (2 * std)
                        
                        current_price = historical_data["close"].iloc[-1]
                        current_lower = lower_band.iloc[-1]
                        
                        # Price below lower band is potential buy signal
                        signal = current_price < current_lower
                        strength = (current_lower - current_price) / current_price if signal else 0
                        indicator_signals[indicator] = signal
                        indicator_confidences[indicator] = min(1.0, strength * 10)
                        reasoning += f"Bollinger Bands: {'Bullish' if signal else 'Bearish'} (Price = {current_price:.2f}, Lower Band = {current_lower:.2f}). "
                
                elif indicator == "Volume":
                    # Volume analysis
                    if len(historical_data) >= 10:
                        avg_volume = historical_data["volume"].rolling(window=10).mean().iloc[-1]
                        current_volume = historical_data["volume"].iloc[-1]
                        price_change = historical_data["close"].pct_change().iloc[-1]
                        
                        # High volume with positive price change is bullish
                        signal = (current_volume > avg_volume * 1.5) and (price_change > 0)
                        strength = (current_volume / avg_volume) * abs(price_change) if signal else 0
                        indicator_signals[indicator] = signal
                        indicator_confidences[indicator] = min(1.0, strength)
                        reasoning += f"Volume: {'Bullish' if signal else 'Bearish'} (Volume = {current_volume:.0f}, Avg = {avg_volume:.0f}, Price Change = {price_change:.2%}). "
                
                elif indicator == "Momentum":
                    # Price momentum
                    if len(historical_data) >= 10:
                        momentum = historical_data["close"].pct_change(periods=10).iloc[-1]
                        signal = momentum > 0.02  # 2% momentum threshold
                        strength = momentum * 10 if signal else 0
                        indicator_signals[indicator] = signal
                        indicator_confidences[indicator] = min(1.0, strength)
                        reasoning += f"Momentum: {'Bullish' if signal else 'Bearish'} (10-day momentum = {momentum:.2%}). "
                
                elif indicator == "Stochastic":
                    # Stochastic Oscillator
                    if len(historical_data) >= 14:
                        low_14 = historical_data["low"].rolling(window=14).min()
                        high_14 = historical_data["high"].rolling(window=14).max()
                        current_close = historical_data["close"].iloc[-1]
                        
                        # Calculate %K
                        k_percent = 100 * ((current_close - low_14.iloc[-1]) / 
                                         (high_14.iloc[-1] - low_14.iloc[-1]) if high_14.iloc[-1] != low_14.iloc[-1] else 0)
                        
                        # Stochastic below 20 is oversold (buy)
                        signal = k_percent < 20 + (30 * profile.risk_tolerance)  # Adjust threshold based on risk tolerance
                        strength = (50 - k_percent) / 30 if k_percent < 50 else 0
                        indicator_signals[indicator] = signal
                        indicator_confidences[indicator] = min(1.0, max(0.0, strength))
                        reasoning += f"Stochastic: {'Bullish' if signal else 'Bearish'} (%K = {k_percent:.2f}). "
            
            # Incorporate model signals if available
            if use_model_signals and model is not None:
                try:
                    # Prepare features for model prediction
                    # This is a simplified implementation
                    features = self._prepare_model_features(historical_data)
                    
                    # Get model prediction
                    prediction = model.predict(features)
                    
                    # Interpret prediction (assuming binary classification: 1 = buy, 0 = don't buy)
                    model_signal = bool(prediction > 0.5)
                    model_confidence = float(abs(prediction - 0.5) * 2)  # Scale to 0-1
                    
                    indicator_signals["Model"] = model_signal
                    indicator_confidences["Model"] = model_confidence
                    indicators_used.append("Model")
                    
                    reasoning += f"Model: {'Bullish' if model_signal else 'Bearish'} (confidence = {model_confidence:.2f}). "
                except Exception as e:
                    logger.warning("Error getting model prediction: %s", e)
            
            # Combine signals based on trader's expertise and risk tolerance
            if indicator_signals:
                # Count positive signals
                positive_signals = sum(1 for signal in indicator_signals.values() if signal)
                total_signals = len(indicator_signals)
                
                # Calculate weighted confidence
                if total_signals > 0:
                    # Weight each indicator by its confidence
                    weighted_confidence = sum(
                        indicator_confidences.get(ind, 0.5) for ind in indicators_used 
                        if indicator_signals.get(ind, False)
                    ) / total_signals
                    
                    # Adjust confidence based on trader's expertise
                    confidence = weighted_confidence * profile.expertise_level
                    
                    # Generate buy signal if enough positive indicators
                    signal_threshold = 0.5 - (0.2 * profile.risk_tolerance)  # More aggressive traders need fewer positive signals
                    buy_signal = (positive_signals / total_signals) >= signal_threshold
                    
                    # Final reasoning
                    reasoning += f"Overall: {positive_signals}/{total_signals} positive signals. "
                    reasoning += f"Decision: {'BUY' if buy_signal else 'HOLD'} with {confidence:.2f} confidence."
        
        except Exception as e:
            logger.error("Error generating buy signal: %s", e)
            reasoning = f"Error in signal generation: {str(e)}"
        
        return buy_signal, confidence, reasoning, indicators_used
    
    def _calculate_exit_confidence(self, 
                                  profile: TraderProfile, 
                                  exit_reason: str, 
                                  pl_pct: float) -> float:
        """Calculate confidence level for an exit decision.
        
        Args:
            profile: Trader profile.
            exit_reason: Reason for exiting the position.
            pl_pct: Profit/loss percentage.
            
        Returns:
            float: Confidence level (0.0 to 1.0).
        """
        if exit_reason == "stop_loss":
            # Higher confidence for larger losses (more urgent to exit)
            return min(1.0, 0.7 + abs(pl_pct) * 2)
        
        elif exit_reason == "profit_target":
            # Higher confidence for larger profits
            return min(1.0, 0.6 + pl_pct)
        
        elif exit_reason == "max_holding_period":
            # Moderate confidence for time-based exits
            return 0.6
        
        else:
            # Default confidence
            return 0.5
    
    def _prepare_model_features(self, historical_data: pd.DataFrame) -> np.ndarray:
        """Prepare features for model prediction.
        
        Args:
            historical_data: Historical price data.
            
        Returns:
            np.ndarray: Features for model prediction.
        """
        # This is a simplified implementation
        # In a real system, this would extract and normalize relevant features
        
        # Example features: returns over different periods, volatility, etc.
        features = []
        
        # Calculate returns
        returns_1d = historical_data["close"].pct_change(periods=1).iloc[-1]
        returns_5d = historical_data["close"].pct_change(periods=5).iloc[-1]
        returns_10d = historical_data["close"].pct_change(periods=10).iloc[-1]
        
        # Calculate volatility
        volatility = historical_data["close"].pct_change().std()
        
        # Calculate volume change
        volume_change = historical_data["volume"].pct_change().iloc[-1]
        
        # Combine features
        features = np.array([returns_1d, returns_5d, returns_10d, volatility, volume_change])
        
        return features.reshape(1, -1)  # Reshape for model input
    
    def compare_trader_performance(self, simulation_ids: List[str]) -> Dict[str, Any]:
        """Compare performance across multiple trader simulations.
        
        Args:
            simulation_ids: List of simulation IDs to compare.
            
        Returns:
            Dict[str, Any]: Comparison results.
            
        Raises:
            ValueError: If any simulation ID doesn't exist.
        """
        # Check if all simulation IDs exist
        for sim_id in simulation_ids:
            if sim_id not in self.simulation_results:
                raise ValueError(f"Simulation ID '{sim_id}' does not exist")
        
        # Extract performance metrics
        comparison = {
            "simulation_ids": simulation_ids,
            "traders": [],
            "total_returns": [],
            "win_rates": [],
            "avg_profit_loss": [],
            "total_trades": []
        }
        
        for sim_id in simulation_ids:
            result = self.simulation_results[sim_id]
            comparison["traders"].append(result["trader_name"])
            comparison["total_returns"].append(result["total_return"])
            comparison["win_rates"].append(result["win_rate"])
            comparison["avg_profit_loss"].append(result["avg_profit_loss"])
            comparison["total_trades"].append(result["total_trades"])
        
        # Find best performer by total return
        best_idx = np.argmax(comparison["total_returns"])
        comparison["best_performer"] = {
            "trader": comparison["traders"][best_idx],
            "total_return": comparison["total_returns"][best_idx],
            "win_rate": comparison["win_rates"][best_idx],
            "avg_profit_loss": comparison["avg_profit_loss"][best_idx],
            "total_trades": comparison["total_trades"][best_idx]
        }
        
        return comparison
    
    def extract_trading_patterns(self, trader_name: str, simulation_id: str = None) -> Dict[str, Any]:
        """Extract trading patterns and insights from a trader's simulation.
        
        Args:
            trader_name: Name of the trader profile.
            simulation_id: Specific simulation ID to analyze (optional).
            
        Returns:
            Dict[str, Any]: Trading patterns and insights.
            
        Raises:
            ValueError: If the trader profile doesn't exist or no simulations are found.
        """
        if trader_name not in self.trader_profiles:
            raise ValueError(f"Trader profile '{trader_name}' does not exist")
        
        # Find relevant simulations
        relevant_sims = []
        for sim_id, result in self.simulation_results.items():
            if result["trader_name"] == trader_name:
                if simulation_id is None or sim_id == simulation_id:
                    relevant_sims.append(result)
        
        if not relevant_sims:
            raise ValueError(f"No simulations found for trader '{trader_name}'")
        
        # Combine trade history from all relevant simulations
        all_trades = []
        for sim in relevant_sims:
            for trade_dict in sim["trade_history"]:
                # Convert dict to TradeDecision object
                if trade_dict["action"] == "buy" and trade_dict["exit_timestamp"] is not None:
                    trade = TradeDecision.from_dict(trade_dict)
                    all_trades.append(trade)
        
        if not all_trades:
            return {"message": "No completed trades found for analysis"}
        
        # Analyze holding periods
        holding_periods = [trade.holding_period for trade in all_trades if trade.holding_period is not None]
        avg_holding_period = sum(holding_periods) / len(holding_periods) if holding_periods else 0
        
        # Analyze profit/loss by exit reason
        pl_by_reason = {}
        count_by_reason = {}
        
        for trade in all_trades:
            if trade.exit_reason and trade.profit_loss_pct is not None:
                if trade.exit_reason not in pl_by_reason:
                    pl_by_reason[trade.exit_reason] = 0
                    count_by_reason[trade.exit_reason] = 0
                
                pl_by_reason[trade.exit_reason] += trade.profit_loss_pct
                count_by_reason[trade.exit_reason] += 1
        
        avg_pl_by_reason = {}
        for reason, total_pl in pl_by_reason.items():
            count = count_by_reason[reason]
            avg_pl_by_reason[reason] = total_pl / count if count > 0 else 0
        
        # Analyze indicators used
        indicator_usage = {}
        indicator_success = {}
        
        for trade in all_trades:
            if trade.indicators_used and trade.profit_loss_pct is not None:
                for indicator in trade.indicators_used:
                    if indicator not in indicator_usage:
                        indicator_usage[indicator] = 0
                        indicator_success[indicator] = 0
                    
                    indicator_usage[indicator] += 1
                    if trade.profit_loss_pct > 0:
                        indicator_success[indicator] += 1
        
        indicator_success_rate = {}
        for indicator, count in indicator_usage.items():
            success = indicator_success[indicator]
            indicator_success_rate[indicator] = success / count if count > 0 else 0
        
        # Compile results
        patterns = {
            "trader_name": trader_name,
            "total_trades_analyzed": len(all_trades),
            "avg_holding_period": avg_holding_period,
            "exit_reasons": {
                "counts": count_by_reason,
                "avg_profit_loss": avg_pl_by_reason
            },
            "indicators": {
                "usage": indicator_usage,
                "success_rate": indicator_success_rate
            },
            "best_indicator": max(indicator_success_rate.items(), key=lambda x: x[1])[0] 
                if indicator_success_rate else None,
            "worst_indicator": min(indicator_success_rate.items(), key=lambda x: x[1])[0] 
                if indicator_success_rate else None,
            "best_exit_reason": max(avg_pl_by_reason.items(), key=lambda x: x[1])[0] 
                if avg_pl_by_reason else None
        }
        
        return patterns