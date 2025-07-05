import logging
import time
from typing import Dict, List, Any, Callable, Optional, TypeVar, Generic, Tuple, Set
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps

logger = logging.getLogger(__name__)

# Type variable for generic batch processing
T = TypeVar('T')
R = TypeVar('R')

class BatchItem(Generic[T, R]):
    """Class representing an item in a batch with its result."""
    
    def __init__(self, item_id: str, data: T):
        """Initialize a batch item.
        
        Args:
            item_id: Unique identifier for the item
            data: The data to process
        """
        self.item_id = item_id
        self.data = data
        self.result: Optional[R] = None
        self.error: Optional[Exception] = None
        self.processed = False
        self.processing_time: Optional[float] = None

class BatchProcessor:
    """Batch processor for efficient handling of portfolio operations.
    
    This class provides functionality for batching operations that would otherwise
    be performed individually, improving performance by reducing overhead and
    enabling parallel processing where appropriate.
    
    Features:
    - Automatic batching of operations
    - Parallel processing capability
    - Batch size optimization
    - Error handling and retry logic
    - Performance metrics tracking
    """
    
    def __init__(self, 
                 max_batch_size: int = 100, 
                 max_wait_time: float = 0.1,
                 use_parallel: bool = True,
                 max_workers: int = 4):
        """Initialize the batch processor.
        
        Args:
            max_batch_size: Maximum number of items in a batch
            max_wait_time: Maximum time to wait for batch to fill in seconds
            use_parallel: Whether to use parallel processing
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.use_parallel = use_parallel
        self.max_workers = max_workers
        
        # Batch storage
        self.current_batch: Dict[str, BatchItem] = {}
        self.batch_start_time = time.time()
        
        # Performance metrics
        self.total_batches_processed = 0
        self.total_items_processed = 0
        self.total_processing_time = 0.0
        self.batch_sizes: List[int] = []
        
        logger.info(f"BatchProcessor initialized with max_batch_size={max_batch_size}, "
                   f"max_wait_time={max_wait_time}s, use_parallel={use_parallel}, "
                   f"max_workers={max_workers}")
    
    def add_item(self, item_id: str, data: T) -> None:
        """Add an item to the current batch.
        
        Args:
            item_id: Unique identifier for the item
            data: The data to process
        """
        if not self.current_batch:
            # This is the first item in a new batch
            self.batch_start_time = time.time()
        
        self.current_batch[item_id] = BatchItem(item_id, data)
        logger.debug(f"Added item {item_id} to batch. Current batch size: {len(self.current_batch)}")
        
        # Process batch if it's full
        if len(self.current_batch) >= self.max_batch_size:
            logger.debug(f"Batch full (size: {len(self.current_batch)}). Processing automatically.")
            return True
        
        # Process batch if max wait time exceeded
        elapsed = time.time() - self.batch_start_time
        if elapsed >= self.max_wait_time and self.current_batch:
            logger.debug(f"Max wait time exceeded ({elapsed:.3f}s). Processing batch.")
            return True
            
        return False
    
    def process_batch(self, processor_func: Callable[[List[T]], Dict[str, R]]) -> Dict[str, R]:
        """Process the current batch using the provided function.
        
        Args:
            processor_func: Function that takes a list of items and returns a dict of results
                           keyed by item_id
        
        Returns:
            Dictionary mapping item_ids to their results
        """
        if not self.current_batch:
            logger.debug("No items in batch to process")
            return {}
        
        batch_size = len(self.current_batch)
        self.batch_sizes.append(batch_size)
        
        # Prepare batch data
        batch_data = {item.item_id: item.data for item in self.current_batch.values()}
        batch_items = list(self.current_batch.values())
        
        start_time = time.time()
        logger.debug(f"Processing batch of {batch_size} items")
        
        try:
            # Process the batch
            if self.use_parallel and batch_size > 1 and self.max_workers > 1:
                results = self._process_parallel(batch_items, processor_func)
            else:
                results = processor_func([item.data for item in batch_items])
            
            # Update batch items with results
            for item_id, result in results.items():
                if item_id in self.current_batch:
                    self.current_batch[item_id].result = result
                    self.current_batch[item_id].processed = True
            
            # Update metrics
            processing_time = time.time() - start_time
            self.total_processing_time += processing_time
            self.total_batches_processed += 1
            self.total_items_processed += batch_size
            
            logger.debug(f"Batch processed in {processing_time:.3f}s")
            
            # Prepare return value and clear batch
            results_copy = {k: v for k, v in results.items()}
            self.current_batch.clear()
            
            return results_copy
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Mark all items as failed
            for item in self.current_batch.values():
                item.error = e
            raise
    
    def _process_parallel(self, 
                          batch_items: List[BatchItem], 
                          processor_func: Callable[[List[T]], Dict[str, R]]) -> Dict[str, R]:
        """Process batch items in parallel.
        
        Args:
            batch_items: List of batch items to process
            processor_func: Function to process the items
            
        Returns:
            Dictionary mapping item_ids to their results
        """
        # Split the batch into smaller chunks for parallel processing
        chunk_size = max(1, len(batch_items) // self.max_workers)
        chunks = [batch_items[i:i + chunk_size] for i in range(0, len(batch_items), chunk_size)]
        
        results = {}
        with ThreadPoolExecutor(max_workers=min(len(chunks), self.max_workers)) as executor:
            # Submit each chunk for processing
            future_to_chunk = {}
            for chunk in chunks:
                chunk_data = [item.data for item in chunk]
                future = executor.submit(processor_func, chunk_data)
                future_to_chunk[future] = chunk
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    chunk_results = future.result()
                    results.update(chunk_results)
                except Exception as e:
                    logger.error(f"Error in parallel chunk processing: {str(e)}")
                    # Mark items in this chunk as failed
                    for item in chunk:
                        item.error = e
                    raise
        
        return results
    
    def flush(self, processor_func: Callable[[List[T]], Dict[str, R]]) -> Dict[str, R]:
        """Force processing of the current batch even if not full.
        
        Args:
            processor_func: Function that takes a list of items and returns a dict of results
                           keyed by item_id
        
        Returns:
            Dictionary mapping item_ids to their results
        """
        return self.process_batch(processor_func)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get batch processing statistics.
        
        Returns:
            Dictionary with batch processing statistics
        """
        avg_batch_size = sum(self.batch_sizes) / len(self.batch_sizes) if self.batch_sizes else 0
        avg_processing_time = (self.total_processing_time / self.total_batches_processed 
                              if self.total_batches_processed > 0 else 0)
        
        return {
            "total_batches_processed": self.total_batches_processed,
            "total_items_processed": self.total_items_processed,
            "total_processing_time": self.total_processing_time,
            "average_batch_size": avg_batch_size,
            "average_processing_time": avg_processing_time,
            "current_batch_size": len(self.current_batch)
        }
    
    def optimize_batch_size(self) -> None:
        """Optimize batch size based on processing history.
        
        This method adjusts the max_batch_size based on historical performance data
        to find the optimal batch size for best throughput.
        """
        if len(self.batch_sizes) < 5:
            logger.debug("Not enough batch history to optimize batch size")
            return
        
        # Simple optimization strategy - adjust based on recent performance
        recent_batch_sizes = self.batch_sizes[-5:]
        avg_recent_size = sum(recent_batch_sizes) / len(recent_batch_sizes)
        
        # If we're consistently hitting max batch size, increase it
        if avg_recent_size >= self.max_batch_size * 0.9:
            new_size = min(int(self.max_batch_size * 1.5), 1000)  # Cap at 1000
            logger.info(f"Optimizing batch size: increasing from {self.max_batch_size} to {new_size}")
            self.max_batch_size = new_size
        # If we're using much smaller batches, decrease max size
        elif avg_recent_size <= self.max_batch_size * 0.5:
            new_size = max(int(self.max_batch_size * 0.8), 10)  # Floor at 10
            logger.info(f"Optimizing batch size: decreasing from {self.max_batch_size} to {new_size}")
            self.max_batch_size = new_size

def batch_operation(batch_processor: BatchProcessor, key_func: Callable = None):
    """Decorator for automatically batching operations.
    
    Args:
        batch_processor: The BatchProcessor instance to use
        key_func: Optional function to extract a key from the first argument
                 If not provided, the first argument is used as the key
    
    Returns:
        Decorated function that will be batched
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Extract key from first argument
            if not args:
                raise ValueError("Batched function must have at least one positional argument")
            
            if key_func is not None:
                item_id = key_func(args[0])
            else:
                item_id = str(args[0])
            
            # Add to batch
            should_process = batch_processor.add_item(item_id, (args, kwargs))
            
            # Define processor function
            def process_batch(batch_items):
                results = {}
                for item_args, item_kwargs in batch_items:
                    result = func(*item_args, **item_kwargs)
                    # Use first arg as key
                    key = key_func(item_args[0]) if key_func else str(item_args[0])
                    results[key] = result
                return results
            
            # Process if needed
            if should_process:
                results = batch_processor.process_batch(process_batch)
                return results.get(item_id)
            
            return None  # Will be processed in a future batch
        
        return wrapper
    
    return decorator