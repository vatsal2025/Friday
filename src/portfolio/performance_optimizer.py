import logging
import time
from typing import Dict, List, Any, Callable, Optional, Set, Tuple, Union
from functools import wraps
import threading
import numpy as np

from .cache_manager import CacheManager
from .batch_processor import BatchProcessor

logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """Coordinator for performance optimization strategies in the portfolio system.
    
    This class integrates various performance optimization techniques including:
    - Caching for expensive calculations and frequent data access
    - Batch processing for efficient operation handling
    - Lazy calculation for deferring non-critical computations
    - Memory optimization for reducing memory footprint
    - Performance monitoring and metrics collection
    
    It provides a unified interface for the portfolio system to leverage these
    optimization strategies.
    """
    
    def __init__(self, 
                 enable_caching: bool = True,
                 enable_batching: bool = True,
                 enable_lazy_calc: bool = True,
                 enable_memory_opt: bool = True,
                 cache_ttl: float = 60.0,
                 max_batch_size: int = 100,
                 max_batch_wait: float = 0.1,
                 use_parallel: bool = True,
                 max_workers: int = 4):
        """Initialize the performance optimizer.
        
        Args:
            enable_caching: Whether to enable caching
            enable_batching: Whether to enable batch processing
            enable_lazy_calc: Whether to enable lazy calculation
            enable_memory_opt: Whether to enable memory optimization
            cache_ttl: Default time-to-live for cache entries in seconds
            max_batch_size: Maximum number of items in a batch
            max_batch_wait: Maximum time to wait for batch to fill in seconds
            use_parallel: Whether to use parallel processing for batches
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.enable_caching = enable_caching
        self.enable_batching = enable_batching
        self.enable_lazy_calc = enable_lazy_calc
        self.enable_memory_opt = enable_memory_opt
        
        # Initialize cache manager if enabled
        self.cache_manager = CacheManager(default_ttl=cache_ttl) if enable_caching else None
        
        # Initialize batch processor if enabled
        self.batch_processor = BatchProcessor(
            max_batch_size=max_batch_size,
            max_wait_time=max_batch_wait,
            use_parallel=use_parallel,
            max_workers=max_workers
        ) if enable_batching else None
        
        # Lazy calculation tracking
        self.pending_calculations: Dict[str, Callable] = {}
        self.calculation_lock = threading.Lock()
        
        # Performance metrics
        self.function_metrics: Dict[str, Dict[str, Any]] = {}
        self.start_time = time.time()
        
        logger.info(f"PerformanceOptimizer initialized with: caching={enable_caching}, "
                   f"batching={enable_batching}, lazy_calc={enable_lazy_calc}, "
                   f"memory_opt={enable_memory_opt}")
    
    def cached(self, ttl: Optional[float] = None):
        """Decorator for caching function results.
        
        Args:
            ttl: Time-to-live for the cached result in seconds
            
        Returns:
            Decorated function or original function if caching disabled
        """
        def decorator(func):
            if not self.enable_caching or self.cache_manager is None:
                return func
            
            return self.cache_manager.cached(ttl)(func)
        return decorator
    
    def batch_operation(self, key_func: Callable = None):
        """Decorator for automatically batching operations.
        
        Args:
            key_func: Optional function to extract a key from the first argument
            
        Returns:
            Decorated function or original function if batching disabled
        """
        from .batch_processor import batch_operation
        
        def decorator(func):
            if not self.enable_batching or self.batch_processor is None:
                return func
            
            return batch_operation(self.batch_processor, key_func)(func)
        return decorator
    
    def lazy_calculation(self, calculation_id: str = None):
        """Decorator for lazy calculation of non-critical values.
        
        Args:
            calculation_id: Optional identifier for the calculation
            
        Returns:
            Decorated function or original function if lazy calculation disabled
        """
        def decorator(func):
            if not self.enable_lazy_calc:
                return func
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                # Generate a calculation ID if not provided
                nonlocal calculation_id
                if calculation_id is None:
                    calculation_id = f"{func.__name__}_{id(args)}_{id(kwargs)}"
                
                # Store the calculation for later execution
                with self.calculation_lock:
                    self.pending_calculations[calculation_id] = lambda: func(*args, **kwargs)
                
                logger.debug(f"Registered lazy calculation: {calculation_id}")
                return None  # Placeholder return value
            
            return wrapper
        return decorator
    
    def execute_pending_calculations(self, max_time: Optional[float] = None) -> int:
        """Execute pending lazy calculations.
        
        Args:
            max_time: Maximum time to spend executing calculations in seconds
                     If None, execute all pending calculations
        
        Returns:
            Number of calculations executed
        """
        if not self.enable_lazy_calc or not self.pending_calculations:
            return 0
        
        start_time = time.time()
        executed_count = 0
        
        # Make a copy of the pending calculations to avoid modification during iteration
        with self.calculation_lock:
            calculations = list(self.pending_calculations.items())
        
        for calc_id, calc_func in calculations:
            # Check if we've exceeded the maximum time
            if max_time is not None and time.time() - start_time > max_time:
                logger.debug(f"Reached max time for lazy calculations. Executed {executed_count} of {len(calculations)}")
                break
            
            try:
                # Execute the calculation
                calc_func()
                executed_count += 1
                
                # Remove from pending list
                with self.calculation_lock:
                    if calc_id in self.pending_calculations:
                        del self.pending_calculations[calc_id]
                
            except Exception as e:
                logger.error(f"Error executing lazy calculation {calc_id}: {str(e)}")
        
        return executed_count
    
    def optimize_memory(self, data: Any) -> Any:
        """Optimize memory usage for the given data.
        
        Args:
            data: Data to optimize
            
        Returns:
            Memory-optimized data
        """
        if not self.enable_memory_opt:
            return data
        
        # Handle different data types
        if isinstance(data, dict):
            return self._optimize_dict(data)
        elif isinstance(data, list):
            return self._optimize_list(data)
        elif isinstance(data, np.ndarray):
            return self._optimize_numpy_array(data)
        else:
            # No optimization for other types
            return data
    
    def _optimize_dict(self, data: Dict) -> Dict:
        """Optimize a dictionary for memory usage.
        
        Args:
            data: Dictionary to optimize
            
        Returns:
            Memory-optimized dictionary
        """
        # Recursively optimize dictionary values
        return {k: self.optimize_memory(v) for k, v in data.items()}
    
    def _optimize_list(self, data: List) -> List:
        """Optimize a list for memory usage.
        
        Args:
            data: List to optimize
            
        Returns:
            Memory-optimized list
        """
        # Check if list can be converted to a more efficient numpy array
        if len(data) > 100 and all(isinstance(x, (int, float)) for x in data):
            return np.array(data, dtype=self._get_optimal_dtype(data))
        
        # Recursively optimize list elements
        return [self.optimize_memory(item) for item in data]
    
    def _optimize_numpy_array(self, data: np.ndarray) -> np.ndarray:
        """Optimize a numpy array for memory usage.
        
        Args:
            data: Numpy array to optimize
            
        Returns:
            Memory-optimized numpy array
        """
        # Downcast to more memory-efficient dtype if possible
        if np.issubdtype(data.dtype, np.floating):
            # Check if float32 is sufficient precision
            if data.dtype == np.float64 and np.all(np.abs(data) < 1e6):
                return data.astype(np.float32)
        elif np.issubdtype(data.dtype, np.integer):
            # Find the smallest integer type that can represent the data
            return data.astype(self._get_optimal_dtype(data))
        
        return data
    
    def _get_optimal_dtype(self, data: Union[List, np.ndarray]) -> np.dtype:
        """Get the optimal numpy dtype for the given data.
        
        Args:
            data: Data to analyze
            
        Returns:
            Optimal numpy dtype
        """
        if isinstance(data, list):
            # Convert to numpy array for analysis
            data = np.array(data)
        
        # Find min and max values
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Determine optimal integer dtype
        if min_val >= 0:
            if max_val <= 255:
                return np.uint8
            elif max_val <= 65535:
                return np.uint16
            elif max_val <= 4294967295:
                return np.uint32
            else:
                return np.uint64
        else:
            if min_val >= -128 and max_val <= 127:
                return np.int8
            elif min_val >= -32768 and max_val <= 32767:
                return np.int16
            elif min_val >= -2147483648 and max_val <= 2147483647:
                return np.int32
            else:
                return np.int64
    
    def monitor_performance(self, category: str = None):
        """Decorator for monitoring function performance.
        
        Args:
            category: Optional category for grouping metrics
            
        Returns:
            Decorated function
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                func_name = func.__name__
                metric_key = f"{category}.{func_name}" if category else func_name
                
                # Initialize metrics for this function if not exists
                if metric_key not in self.function_metrics:
                    self.function_metrics[metric_key] = {
                        "calls": 0,
                        "total_time": 0.0,
                        "min_time": float('inf'),
                        "max_time": 0.0,
                        "last_call_time": None
                    }
                
                # Measure execution time
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                
                # Update metrics
                metrics = self.function_metrics[metric_key]
                metrics["calls"] += 1
                metrics["total_time"] += execution_time
                metrics["min_time"] = min(metrics["min_time"], execution_time)
                metrics["max_time"] = max(metrics["max_time"], execution_time)
                metrics["last_call_time"] = time.time()
                
                # Log if execution time is unusually long
                avg_time = metrics["total_time"] / metrics["calls"]
                if execution_time > avg_time * 2 and metrics["calls"] > 5:
                    logger.warning(f"Slow execution of {metric_key}: {execution_time:.3f}s "
                                  f"(avg: {avg_time:.3f}s)")
                
                return result
            
            return wrapper
        return decorator
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            "uptime": time.time() - self.start_time,
            "function_metrics": self._get_function_metrics_summary(),
            "pending_lazy_calculations": len(self.pending_calculations)
        }
        
        # Add cache metrics if enabled
        if self.enable_caching and self.cache_manager is not None:
            metrics["cache"] = self.cache_manager.get_stats()
        
        # Add batch metrics if enabled
        if self.enable_batching and self.batch_processor is not None:
            metrics["batch"] = self.batch_processor.get_stats()
        
        return metrics
    
    def _get_function_metrics_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get a summary of function performance metrics.
        
        Returns:
            Dictionary with function metrics summary
        """
        summary = {}
        
        for func_name, metrics in self.function_metrics.items():
            if metrics["calls"] == 0:
                continue
                
            avg_time = metrics["total_time"] / metrics["calls"]
            summary[func_name] = {
                "calls": metrics["calls"],
                "total_time": metrics["total_time"],
                "avg_time": avg_time,
                "min_time": metrics["min_time"],
                "max_time": metrics["max_time"]
            }
        
        return summary
    
    def reset_metrics(self) -> None:
        """Reset all performance metrics."""
        self.function_metrics.clear()
        self.start_time = time.time()
        
        if self.enable_caching and self.cache_manager is not None:
            self.cache_manager.hits = 0
            self.cache_manager.misses = 0
            self.cache_manager.invalidations = 0
        
        if self.enable_batching and self.batch_processor is not None:
            self.batch_processor.total_batches_processed = 0
            self.batch_processor.total_items_processed = 0
            self.batch_processor.total_processing_time = 0.0
            self.batch_processor.batch_sizes = []
        
        logger.info("Performance metrics reset")
    
    def optimize_batch_sizes(self) -> None:
        """Optimize batch sizes based on historical performance."""
        if self.enable_batching and self.batch_processor is not None:
            self.batch_processor.optimize_batch_size()
    
    def cleanup(self) -> None:
        """Perform cleanup operations like clearing expired cache entries."""
        if self.enable_caching and self.cache_manager is not None:
            removed = self.cache_manager.cleanup_expired()
            logger.debug(f"Cleaned up {removed} expired cache entries")
        
        # Execute any critical pending calculations
        if self.enable_lazy_calc and self.pending_calculations:
            executed = self.execute_pending_calculations(max_time=0.1)
            logger.debug(f"Executed {executed} pending calculations during cleanup")