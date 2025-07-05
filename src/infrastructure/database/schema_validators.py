"""MongoDB schema validators for the Friday AI Trading System.

This module provides JSON Schema validators for MongoDB collections to ensure data integrity.
"""

from typing import Dict, Any

# Market Data Schema Validator
MARKET_DATA_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["symbol", "exchange", "timeframe", "data"],
        "properties": {
            "symbol": {
                "bsonType": "string",
                "description": "The trading symbol, required"
            },
            "exchange": {
                "bsonType": "string",
                "description": "The exchange name, required"
            },
            "timeframe": {
                "bsonType": "string",
                "description": "The timeframe of the data, required",
                "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
            },
            "data": {
                "bsonType": "array",
                "description": "Array of OHLCV data points, required",
                "items": {
                    "bsonType": "object",
                    "required": ["timestamp", "open", "high", "low", "close", "volume"],
                    "properties": {
                        "timestamp": {
                            "bsonType": "date",
                            "description": "The timestamp of the data point, required"
                        },
                        "open": {
                            "bsonType": "double",
                            "description": "The opening price, required"
                        },
                        "high": {
                            "bsonType": "double",
                            "description": "The highest price, required"
                        },
                        "low": {
                            "bsonType": "double",
                            "description": "The lowest price, required"
                        },
                        "close": {
                            "bsonType": "double",
                            "description": "The closing price, required"
                        },
                        "volume": {
                            "bsonType": "double",
                            "description": "The trading volume, required"
                        }
                    }
                }
            },
            "metadata": {
                "bsonType": "object",
                "description": "Additional metadata for the market data"
            },
            "created_at": {
                "bsonType": "date",
                "description": "The creation timestamp"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "The last update timestamp"
            }
        }
    }
}

# Tick Data Schema Validator
TICK_DATA_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["symbol", "exchange", "timestamp", "price", "volume"],
        "properties": {
            "symbol": {
                "bsonType": "string",
                "description": "The trading symbol, required"
            },
            "exchange": {
                "bsonType": "string",
                "description": "The exchange name, required"
            },
            "timestamp": {
                "bsonType": "date",
                "description": "The timestamp of the tick, required"
            },
            "price": {
                "bsonType": "double",
                "description": "The price of the tick, required"
            },
            "volume": {
                "bsonType": "double",
                "description": "The volume of the tick, required"
            },
            "bid": {
                "bsonType": "double",
                "description": "The bid price"
            },
            "ask": {
                "bsonType": "double",
                "description": "The ask price"
            },
            "trade_id": {
                "bsonType": "string",
                "description": "The trade ID"
            },
            "metadata": {
                "bsonType": "object",
                "description": "Additional metadata for the tick data"
            },
            "created_at": {
                "bsonType": "date",
                "description": "The creation timestamp"
            }
        }
    }
}

# Order Book Schema Validator
ORDER_BOOK_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["symbol", "exchange", "timestamp", "bids", "asks"],
        "properties": {
            "symbol": {
                "bsonType": "string",
                "description": "The trading symbol, required"
            },
            "exchange": {
                "bsonType": "string",
                "description": "The exchange name, required"
            },
            "timestamp": {
                "bsonType": "date",
                "description": "The timestamp of the order book snapshot, required"
            },
            "bids": {
                "bsonType": "array",
                "description": "Array of bid entries, required",
                "items": {
                    "bsonType": "object",
                    "required": ["price", "volume"],
                    "properties": {
                        "price": {
                            "bsonType": "double",
                            "description": "The bid price, required"
                        },
                        "volume": {
                            "bsonType": "double",
                            "description": "The bid volume, required"
                        }
                    }
                }
            },
            "asks": {
                "bsonType": "array",
                "description": "Array of ask entries, required",
                "items": {
                    "bsonType": "object",
                    "required": ["price", "volume"],
                    "properties": {
                        "price": {
                            "bsonType": "double",
                            "description": "The ask price, required"
                        },
                        "volume": {
                            "bsonType": "double",
                            "description": "The ask volume, required"
                        }
                    }
                }
            },
            "metadata": {
                "bsonType": "object",
                "description": "Additional metadata for the order book"
            },
            "created_at": {
                "bsonType": "date",
                "description": "The creation timestamp"
            }
        }
    }
}

# Model Storage Schema Validator
MODEL_STORAGE_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["name", "version", "model_type", "framework", "created_at"],
        "properties": {
            "name": {
                "bsonType": "string",
                "description": "The name of the model, required"
            },
            "version": {
                "bsonType": "string",
                "description": "The version of the model, required"
            },
            "model_type": {
                "bsonType": "string",
                "description": "The type of the model, required",
                "enum": ["classification", "regression", "clustering", "reinforcement_learning", "time_series", "nlp", "custom"]
            },
            "framework": {
                "bsonType": "string",
                "description": "The framework used to build the model, required",
                "enum": ["tensorflow", "pytorch", "scikit_learn", "xgboost", "lightgbm", "keras", "custom"]
            },
            "binary_data": {
                "bsonType": "binData",
                "description": "The binary data of the model"
            },
            "file_path": {
                "bsonType": "string",
                "description": "The file path where the model is stored"
            },
            "metrics": {
                "bsonType": "object",
                "description": "The performance metrics of the model",
                "properties": {
                    "accuracy": {
                        "bsonType": "double",
                        "description": "The accuracy of the model"
                    },
                    "precision": {
                        "bsonType": "double",
                        "description": "The precision of the model"
                    },
                    "recall": {
                        "bsonType": "double",
                        "description": "The recall of the model"
                    },
                    "f1_score": {
                        "bsonType": "double",
                        "description": "The F1 score of the model"
                    },
                    "mse": {
                        "bsonType": "double",
                        "description": "The mean squared error of the model"
                    },
                    "mae": {
                        "bsonType": "double",
                        "description": "The mean absolute error of the model"
                    },
                    "r2": {
                        "bsonType": "double",
                        "description": "The R-squared score of the model"
                    },
                    "custom_metrics": {
                        "bsonType": "object",
                        "description": "Custom metrics for the model"
                    }
                }
            },
            "hyperparameters": {
                "bsonType": "object",
                "description": "The hyperparameters used to train the model"
            },
            "features": {
                "bsonType": "array",
                "description": "The features used to train the model",
                "items": {
                    "bsonType": "string"
                }
            },
            "target": {
                "bsonType": "string",
                "description": "The target variable for the model"
            },
            "training_data": {
                "bsonType": "object",
                "description": "Information about the training data",
                "properties": {
                    "start_date": {
                        "bsonType": "date",
                        "description": "The start date of the training data"
                    },
                    "end_date": {
                        "bsonType": "date",
                        "description": "The end date of the training data"
                    },
                    "symbols": {
                        "bsonType": "array",
                        "description": "The symbols used in the training data",
                        "items": {
                            "bsonType": "string"
                        }
                    },
                    "timeframe": {
                        "bsonType": "string",
                        "description": "The timeframe of the training data"
                    },
                    "data_points": {
                        "bsonType": "int",
                        "description": "The number of data points in the training data"
                    }
                }
            },
            "created_at": {
                "bsonType": "date",
                "description": "The creation timestamp, required"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "The last update timestamp"
            },
            "metadata": {
                "bsonType": "object",
                "description": "Additional metadata for the model"
            }
        }
    }
}

# Trading Strategy Schema Validator
TRADING_STRATEGY_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["name", "description", "created_at"],
        "properties": {
            "name": {
                "bsonType": "string",
                "description": "The name of the strategy, required"
            },
            "description": {
                "bsonType": "string",
                "description": "The description of the strategy, required"
            },
            "code": {
                "bsonType": "string",
                "description": "The code implementation of the strategy"
            },
            "parameters": {
                "bsonType": "object",
                "description": "The parameters of the strategy"
            },
            "performance": {
                "bsonType": "object",
                "description": "The performance metrics of the strategy",
                "properties": {
                    "sharpe_ratio": {
                        "bsonType": "double",
                        "description": "The Sharpe ratio of the strategy"
                    },
                    "sortino_ratio": {
                        "bsonType": "double",
                        "description": "The Sortino ratio of the strategy"
                    },
                    "max_drawdown": {
                        "bsonType": "double",
                        "description": "The maximum drawdown of the strategy"
                    },
                    "win_rate": {
                        "bsonType": "double",
                        "description": "The win rate of the strategy"
                    },
                    "profit_factor": {
                        "bsonType": "double",
                        "description": "The profit factor of the strategy"
                    },
                    "annual_return": {
                        "bsonType": "double",
                        "description": "The annual return of the strategy"
                    },
                    "custom_metrics": {
                        "bsonType": "object",
                        "description": "Custom performance metrics for the strategy"
                    }
                }
            },
            "backtest_results": {
                "bsonType": "array",
                "description": "The backtest results of the strategy",
                "items": {
                    "bsonType": "object",
                    "properties": {
                        "start_date": {
                            "bsonType": "date",
                            "description": "The start date of the backtest"
                        },
                        "end_date": {
                            "bsonType": "date",
                            "description": "The end date of the backtest"
                        },
                        "symbols": {
                            "bsonType": "array",
                            "description": "The symbols used in the backtest",
                            "items": {
                                "bsonType": "string"
                            }
                        },
                        "timeframe": {
                            "bsonType": "string",
                            "description": "The timeframe of the backtest"
                        },
                        "initial_capital": {
                            "bsonType": "double",
                            "description": "The initial capital for the backtest"
                        },
                        "final_capital": {
                            "bsonType": "double",
                            "description": "The final capital after the backtest"
                        },
                        "total_trades": {
                            "bsonType": "int",
                            "description": "The total number of trades in the backtest"
                        },
                        "winning_trades": {
                            "bsonType": "int",
                            "description": "The number of winning trades in the backtest"
                        },
                        "losing_trades": {
                            "bsonType": "int",
                            "description": "The number of losing trades in the backtest"
                        },
                        "performance_metrics": {
                            "bsonType": "object",
                            "description": "The performance metrics of the backtest"
                        },
                        "trades": {
                            "bsonType": "array",
                            "description": "The trades executed in the backtest",
                            "items": {
                                "bsonType": "object"
                            }
                        }
                    }
                }
            },
            "created_at": {
                "bsonType": "date",
                "description": "The creation timestamp, required"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "The last update timestamp"
            },
            "metadata": {
                "bsonType": "object",
                "description": "Additional metadata for the strategy"
            }
        }
    }
}

# Backtest Results Schema Validator
BACKTEST_RESULTS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["strategy_id", "start_date", "end_date", "symbols", "created_at"],
        "properties": {
            "strategy_id": {
                "bsonType": "string",
                "description": "The ID of the strategy, required"
            },
            "start_date": {
                "bsonType": "date",
                "description": "The start date of the backtest, required"
            },
            "end_date": {
                "bsonType": "date",
                "description": "The end date of the backtest, required"
            },
            "symbols": {
                "bsonType": "array",
                "description": "The symbols used in the backtest, required",
                "items": {
                    "bsonType": "string"
                }
            },
            "timeframe": {
                "bsonType": "string",
                "description": "The timeframe of the backtest",
                "enum": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"]
            },
            "initial_capital": {
                "bsonType": "double",
                "description": "The initial capital for the backtest"
            },
            "final_capital": {
                "bsonType": "double",
                "description": "The final capital after the backtest"
            },
            "total_trades": {
                "bsonType": "int",
                "description": "The total number of trades in the backtest"
            },
            "winning_trades": {
                "bsonType": "int",
                "description": "The number of winning trades in the backtest"
            },
            "losing_trades": {
                "bsonType": "int",
                "description": "The number of losing trades in the backtest"
            },
            "performance": {
                "bsonType": "object",
                "description": "The performance metrics of the backtest",
                "properties": {
                    "sharpe_ratio": {
                        "bsonType": "double",
                        "description": "The Sharpe ratio"
                    },
                    "sortino_ratio": {
                        "bsonType": "double",
                        "description": "The Sortino ratio"
                    },
                    "max_drawdown": {
                        "bsonType": "double",
                        "description": "The maximum drawdown"
                    },
                    "win_rate": {
                        "bsonType": "double",
                        "description": "The win rate"
                    },
                    "profit_factor": {
                        "bsonType": "double",
                        "description": "The profit factor"
                    },
                    "annual_return": {
                        "bsonType": "double",
                        "description": "The annual return"
                    },
                    "total_return": {
                        "bsonType": "double",
                        "description": "The total return"
                    },
                    "volatility": {
                        "bsonType": "double",
                        "description": "The volatility"
                    },
                    "custom_metrics": {
                        "bsonType": "object",
                        "description": "Custom performance metrics"
                    }
                }
            },
            "trades": {
                "bsonType": "array",
                "description": "The trades executed in the backtest",
                "items": {
                    "bsonType": "object",
                    "properties": {
                        "trade_id": {
                            "bsonType": "string",
                            "description": "The trade ID"
                        },
                        "symbol": {
                            "bsonType": "string",
                            "description": "The trading symbol"
                        },
                        "side": {
                            "bsonType": "string",
                            "description": "The trade side",
                            "enum": ["buy", "sell"]
                        },
                        "quantity": {
                            "bsonType": "double",
                            "description": "The trade quantity"
                        },
                        "entry_price": {
                            "bsonType": "double",
                            "description": "The entry price"
                        },
                        "exit_price": {
                            "bsonType": "double",
                            "description": "The exit price"
                        },
                        "entry_timestamp": {
                            "bsonType": "date",
                            "description": "The entry timestamp"
                        },
                        "exit_timestamp": {
                            "bsonType": "date",
                            "description": "The exit timestamp"
                        },
                        "pnl": {
                            "bsonType": "double",
                            "description": "The profit and loss"
                        },
                        "commission": {
                            "bsonType": "double",
                            "description": "The commission paid"
                        }
                    }
                }
            },
            "parameters": {
                "bsonType": "object",
                "description": "The strategy parameters used in the backtest"
            },
            "created_at": {
                "bsonType": "date",
                "description": "The creation timestamp, required"
            },
            "metadata": {
                "bsonType": "object",
                "description": "Additional metadata for the backtest"
            }
        }
    }
}

# Trading Signals Schema Validator
TRADING_SIGNALS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["symbol", "exchange", "strategy_id", "signal_type", "timestamp", "created_at"],
        "properties": {
            "symbol": {
                "bsonType": "string",
                "description": "The trading symbol, required"
            },
            "exchange": {
                "bsonType": "string",
                "description": "The exchange name, required"
            },
            "strategy_id": {
                "bsonType": "string",
                "description": "The ID of the strategy that generated the signal, required"
            },
            "signal_type": {
                "bsonType": "string",
                "description": "The type of signal, required",
                "enum": ["buy", "sell", "hold", "entry", "exit", "stop_loss", "take_profit"]
            },
            "timestamp": {
                "bsonType": "date",
                "description": "The timestamp when the signal was generated, required"
            },
            "confidence": {
                "bsonType": "double",
                "description": "The confidence level of the signal (0.0 to 1.0)",
                "minimum": 0.0,
                "maximum": 1.0
            },
            "price": {
                "bsonType": "double",
                "description": "The price at which the signal was generated"
            },
            "volume": {
                "bsonType": "double",
                "description": "The suggested volume for the signal"
            },
            "stop_loss": {
                "bsonType": "double",
                "description": "The suggested stop loss price"
            },
            "take_profit": {
                "bsonType": "double",
                "description": "The suggested take profit price"
            },
            "expiry": {
                "bsonType": "date",
                "description": "The expiry timestamp for the signal"
            },
            "indicators": {
                "bsonType": "object",
                "description": "Technical indicators that triggered the signal"
            },
            "metadata": {
                "bsonType": "object",
                "description": "Additional metadata for the signal"
            },
            "created_at": {
                "bsonType": "date",
                "description": "The creation timestamp, required"
            }
        }
    }
}

# Trading Orders Schema Validator
TRADING_ORDERS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["symbol", "exchange", "strategy_id", "order_id", "order_type", "side", "quantity", "status", "timestamp", "created_at"],
        "properties": {
            "symbol": {
                "bsonType": "string",
                "description": "The trading symbol, required"
            },
            "exchange": {
                "bsonType": "string",
                "description": "The exchange name, required"
            },
            "strategy_id": {
                "bsonType": "string",
                "description": "The ID of the strategy that created the order, required"
            },
            "order_id": {
                "bsonType": "string",
                "description": "The unique order ID, required"
            },
            "order_type": {
                "bsonType": "string",
                "description": "The type of order, required",
                "enum": ["market", "limit", "stop", "stop_limit", "trailing_stop"]
            },
            "side": {
                "bsonType": "string",
                "description": "The order side, required",
                "enum": ["buy", "sell"]
            },
            "quantity": {
                "bsonType": "double",
                "description": "The order quantity, required"
            },
            "price": {
                "bsonType": "double",
                "description": "The order price (for limit orders)"
            },
            "stop_price": {
                "bsonType": "double",
                "description": "The stop price (for stop orders)"
            },
            "filled_quantity": {
                "bsonType": "double",
                "description": "The quantity that has been filled"
            },
            "remaining_quantity": {
                "bsonType": "double",
                "description": "The remaining quantity to be filled"
            },
            "average_fill_price": {
                "bsonType": "double",
                "description": "The average price at which the order was filled"
            },
            "status": {
                "bsonType": "string",
                "description": "The order status, required",
                "enum": ["pending", "open", "partially_filled", "filled", "cancelled", "rejected", "expired"]
            },
            "timestamp": {
                "bsonType": "date",
                "description": "The timestamp when the order was placed, required"
            },
            "filled_timestamp": {
                "bsonType": "date",
                "description": "The timestamp when the order was filled"
            },
            "cancelled_timestamp": {
                "bsonType": "date",
                "description": "The timestamp when the order was cancelled"
            },
            "time_in_force": {
                "bsonType": "string",
                "description": "The time in force for the order",
                "enum": ["GTC", "IOC", "FOK", "DAY"]
            },
            "commission": {
                "bsonType": "double",
                "description": "The commission paid for the order"
            },
            "fills": {
                "bsonType": "array",
                "description": "Array of order fills",
                "items": {
                    "bsonType": "object",
                    "properties": {
                        "fill_id": {
                            "bsonType": "string",
                            "description": "The fill ID"
                        },
                        "quantity": {
                            "bsonType": "double",
                            "description": "The filled quantity"
                        },
                        "price": {
                            "bsonType": "double",
                            "description": "The fill price"
                        },
                        "timestamp": {
                            "bsonType": "date",
                            "description": "The fill timestamp"
                        },
                        "commission": {
                            "bsonType": "double",
                            "description": "The commission for this fill"
                        }
                    }
                }
            },
            "metadata": {
                "bsonType": "object",
                "description": "Additional metadata for the order"
            },
            "created_at": {
                "bsonType": "date",
                "description": "The creation timestamp, required"
            }
        }
    }
}

# Trading Positions Schema Validator
TRADING_POSITIONS_VALIDATOR = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["symbol", "exchange", "strategy_id", "position_id", "side", "quantity", "status", "open_timestamp", "created_at"],
        "properties": {
            "symbol": {
                "bsonType": "string",
                "description": "The trading symbol, required"
            },
            "exchange": {
                "bsonType": "string",
                "description": "The exchange name, required"
            },
            "strategy_id": {
                "bsonType": "string",
                "description": "The ID of the strategy that opened the position, required"
            },
            "position_id": {
                "bsonType": "string",
                "description": "The unique position ID, required"
            },
            "side": {
                "bsonType": "string",
                "description": "The position side, required",
                "enum": ["long", "short"]
            },
            "quantity": {
                "bsonType": "double",
                "description": "The position quantity, required"
            },
            "entry_price": {
                "bsonType": "double",
                "description": "The average entry price"
            },
            "exit_price": {
                "bsonType": "double",
                "description": "The average exit price"
            },
            "current_price": {
                "bsonType": "double",
                "description": "The current price of the position"
            },
            "unrealized_pnl": {
                "bsonType": "double",
                "description": "The unrealized profit and loss"
            },
            "realized_pnl": {
                "bsonType": "double",
                "description": "The realized profit and loss"
            },
            "stop_loss": {
                "bsonType": "double",
                "description": "The stop loss price"
            },
            "take_profit": {
                "bsonType": "double",
                "description": "The take profit price"
            },
            "status": {
                "bsonType": "string",
                "description": "The position status, required",
                "enum": ["open", "closed", "partially_closed"]
            },
            "open_timestamp": {
                "bsonType": "date",
                "description": "The timestamp when the position was opened, required"
            },
            "close_timestamp": {
                "bsonType": "date",
                "description": "The timestamp when the position was closed"
            },
            "orders": {
                "bsonType": "array",
                "description": "Array of order IDs associated with this position",
                "items": {
                    "bsonType": "string"
                }
            },
            "commission": {
                "bsonType": "double",
                "description": "The total commission paid for the position"
            },
            "metadata": {
                "bsonType": "object",
                "description": "Additional metadata for the position"
            },
            "created_at": {
                "bsonType": "date",
                "description": "The creation timestamp, required"
            },
            "updated_at": {
                "bsonType": "date",
                "description": "The last update timestamp"
            }
        }
    }
}


def get_validator(collection_type: str) -> Dict[str, Any]:
    """Get the schema validator for a specific collection type.

    Args:
        collection_type: The type of collection to get the validator for.

    Returns:
        Dict[str, Any]: The schema validator for the collection type.

    Raises:
        ValueError: If the collection type is not supported.
    """
    validators = {
        "market_data": MARKET_DATA_VALIDATOR,
        "tick_data": TICK_DATA_VALIDATOR,
        "order_book": ORDER_BOOK_VALIDATOR,
        "model_storage": MODEL_STORAGE_VALIDATOR,
        "trading_strategy": TRADING_STRATEGY_VALIDATOR,
        "backtest_results": BACKTEST_RESULTS_VALIDATOR,
        "trading_signals": TRADING_SIGNALS_VALIDATOR,
        "trading_orders": TRADING_ORDERS_VALIDATOR,
        "trading_positions": TRADING_POSITIONS_VALIDATOR,
    }

    if collection_type not in validators:
        raise ValueError(f"Unsupported collection type: {collection_type}")

    return validators[collection_type]
