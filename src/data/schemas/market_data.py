"""Market data schemas for the Friday AI Trading System.

This module defines the schemas for market data used in the system.
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class TimeFrame(str, Enum):
    """Time frame for market data."""
    
    ONE_MINUTE = "1m"
    FIVE_MINUTES = "5m"
    FIFTEEN_MINUTES = "15m"
    THIRTY_MINUTES = "30m"
    ONE_HOUR = "1h"
    FOUR_HOURS = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


class OHLCV(BaseModel):
    """Open, High, Low, Close, Volume data."""
    
    timestamp: datetime = Field(..., description="Timestamp of the data point")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="Highest price")
    low: float = Field(..., description="Lowest price")
    close: float = Field(..., description="Closing price")
    volume: float = Field(..., description="Trading volume")
    
    @validator('timestamp')
    def timestamp_must_be_utc(cls, v):
        """Validate that timestamp is in UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=datetime.timezone.utc)
        return v
    
    class Config:
        """Pydantic model configuration."""
        
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class MarketData(BaseModel):
    """Market data for a symbol."""
    
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange where the symbol is traded")
    timeframe: TimeFrame = Field(..., description="Time frame of the data")
    data: List[OHLCV] = Field(..., description="OHLCV data points")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update time")
    metadata: Dict[str, Union[str, int, float, bool]] = Field(default_factory=dict, description="Additional metadata")


class TickData(BaseModel):
    """Tick-by-tick market data."""
    
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange where the symbol is traded")
    timestamp: datetime = Field(..., description="Timestamp of the tick")
    price: float = Field(..., description="Price of the tick")
    volume: float = Field(..., description="Volume of the tick")
    bid: Optional[float] = Field(None, description="Bid price")
    ask: Optional[float] = Field(None, description="Ask price")
    bid_volume: Optional[float] = Field(None, description="Bid volume")
    ask_volume: Optional[float] = Field(None, description="Ask volume")
    
    @validator('timestamp')
    def timestamp_must_be_utc(cls, v):
        """Validate that timestamp is in UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=datetime.timezone.utc)
        return v
    
    class Config:
        """Pydantic model configuration."""
        
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }


class OrderBookEntry(BaseModel):
    """Entry in an order book."""
    
    price: float = Field(..., description="Price level")
    volume: float = Field(..., description="Volume at this price level")


class OrderBook(BaseModel):
    """Order book for a symbol."""
    
    symbol: str = Field(..., description="Trading symbol")
    exchange: str = Field(..., description="Exchange where the symbol is traded")
    timestamp: datetime = Field(..., description="Timestamp of the order book")
    bids: List[OrderBookEntry] = Field(..., description="Bid entries")
    asks: List[OrderBookEntry] = Field(..., description="Ask entries")
    
    @validator('timestamp')
    def timestamp_must_be_utc(cls, v):
        """Validate that timestamp is in UTC."""
        if v.tzinfo is None:
            return v.replace(tzinfo=datetime.timezone.utc)
        return v
    
    class Config:
        """Pydantic model configuration."""
        
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }