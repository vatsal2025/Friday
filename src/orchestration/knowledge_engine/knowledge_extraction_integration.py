import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import concurrent.futures

# Import necessary infrastructure components
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.database import mongodb
from src.infrastructure.event.event_system import EventSystem
from src.infrastructure.logging import get_logger

# Import knowledge engine components
from src.orchestration.knowledge_engine.book_knowledge_extractor import BookKnowledgeExtractor
from src.orchestration.knowledge_engine.knowledge_base_builder import KnowledgeBaseBuilder
from src.orchestration.knowledge_engine.strategy_generator import StrategyGenerator
from src.orchestration.knowledge_engine.multimodal_content_processor import MultimodalContentProcessor
from src.orchestration.knowledge_engine.ocr_book_digitizer import OCRBookDigitizer

class KnowledgeExtractionIntegration:
    """
    Main orchestrator for the Knowledge Extraction System.
    
    This component integrates all the knowledge extraction components and provides
    a unified interface for the entire knowledge extraction pipeline, from book
    digitization to strategy generation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Knowledge Extraction Integration with configuration settings.
        
        Args:
            config: Optional configuration dictionary. If None, loads from unified_config.
        """
        self.config_manager = ConfigManager()
        self.config = config if config else self.config_manager.get_config('knowledge_extraction')
        
        # Initialize database connection
        self.extraction_jobs_collection = mongodb.get_collection('knowledge_extraction_jobs')
        
        # Initialize event system
        self.event_system = EventSystem()
        
        # Configure logging
        self.logger = get_logger(__name__)
        
        # Initialize component instances
        self._initialize_components()
        
        self.logger.info("Knowledge Extraction Integration initialized")
    
    def _initialize_components(self):
        """
        Initialize all knowledge extraction components.
        """
        # Initialize OCR Book Digitizer
        self.ocr_digitizer = OCRBookDigitizer(self.config)
        
        # Initialize Multimodal Content Processor
        self.content_processor = MultimodalContentProcessor(self.config)
        
        # Initialize Book Knowledge Extractor
        self.knowledge_extractor = BookKnowledgeExtractor(self.config)
        
        # Initialize Knowledge Base Builder
        self.knowledge_base = KnowledgeBaseBuilder(self.config)
        
        # Initialize Strategy Generator
        self.strategy_generator = StrategyGenerator(self.config)
        
        self.logger.info("All knowledge extraction components initialized")
    
    def process_book(self, book_path: str, metadata: Optional[Dict[str, Any]] = None, 
                   options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Process a book through the entire knowledge extraction pipeline.
        
        Args:
            book_path: Path to the book file or directory
            metadata: Optional metadata about the book
            options: Optional processing options
            
        Returns:
            Dictionary with processing results
        """
        job_id = self._create_extraction_job(book_path, metadata, options)
        
        try:
            self.logger.info(f"Starting knowledge extraction pipeline for book: {book_path}")
            
            # Step 1: Digitize the book if needed
            digitization_result = self._digitize_book(book_path, metadata, job_id)
            if not digitization_result['success']:
                self._update_job_status(job_id, 'failed', digitization_result['error'])
                return {'success': False, 'error': digitization_result['error'], 'job_id': job_id}
            
            book_id = digitization_result['book_id']
            self._update_job_status(job_id, 'digitized', f"Book digitized with ID: {book_id}")
            
            # Step 2: Process multimodal content
            processing_result = self._process_content(book_id, job_id)
            if not processing_result['success']:
                self._update_job_status(job_id, 'failed', processing_result['error'])
                return {'success': False, 'error': processing_result['error'], 'job_id': job_id}
            
            content_ids = processing_result['content_ids']
            self._update_job_status(job_id, 'content_processed', 
                                  f"Processed {len(content_ids)} content items")
            
            # Step 3: Extract knowledge
            extraction_result = self._extract_knowledge(book_id, content_ids, job_id)
            if not extraction_result['success']:
                self._update_job_status(job_id, 'failed', extraction_result['error'])
                return {'success': False, 'error': extraction_result['error'], 'job_id': job_id}
            
            knowledge_items = extraction_result['knowledge_items']
            self._update_job_status(job_id, 'knowledge_extracted', 
                                  f"Extracted {len(knowledge_items)} knowledge items")
            
            # Step 4: Build knowledge base
            kb_result = self._build_knowledge_base(knowledge_items, book_id, job_id)
            if not kb_result['success']:
                self._update_job_status(job_id, 'failed', kb_result['error'])
                return {'success': False, 'error': kb_result['error'], 'job_id': job_id}
            
            kb_ids = kb_result['knowledge_base_ids']
            self._update_job_status(job_id, 'knowledge_base_built', 
                                  f"Added {len(kb_ids)} items to knowledge base")
            
            # Step 5: Generate strategies (if enabled)
            if options and options.get('generate_strategies', False):
                strategy_result = self._generate_strategies(kb_ids, job_id)
                if not strategy_result['success']:
                    self._update_job_status(job_id, 'failed', strategy_result['error'])
                    return {'success': False, 'error': strategy_result['error'], 'job_id': job_id}
                
                strategy_ids = strategy_result['strategy_ids']
                self._update_job_status(job_id, 'strategies_generated', 
                                      f"Generated {len(strategy_ids)} strategies")
            
            # Complete the job
            self._update_job_status(job_id, 'completed', "Knowledge extraction pipeline completed")
            
            # Prepare final result
            result = {
                'success': True,
                'job_id': job_id,
                'book_id': book_id,
                'content_ids': content_ids,
                'knowledge_items_count': len(knowledge_items),
                'knowledge_base_ids': kb_ids
            }
            
            if options and options.get('generate_strategies', False):
                result['strategy_ids'] = strategy_result.get('strategy_ids', [])
            
            # Publish completion event
            self.event_system.publish('knowledge_extraction_completed', {
                'job_id': job_id,
                'book_id': book_id,
                'knowledge_items_count': len(knowledge_items),
                'timestamp': datetime.now()
            })
            
            return result
        
        except Exception as e:
            error_msg = f"Error in knowledge extraction pipeline: {str(e)}"
            self.logger.error(error_msg)
            self._update_job_status(job_id, 'failed', error_msg)
            return {'success': False, 'error': error_msg, 'job_id': job_id}
    
    def _create_extraction_job(self, book_path: str, metadata: Optional[Dict[str, Any]], 
                             options: Optional[Dict[str, Any]]) -> str:
        """
        Create a new knowledge extraction job.
        
        Args:
            book_path: Path to the book
            metadata: Book metadata
            options: Processing options
            
        Returns:
            Job ID
        """
        job = {
            'book_path': book_path,
            'metadata': metadata or {},
            'options': options or {},
            'status': 'created',
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'progress': 0,
            'steps': [
                {'name': 'digitization', 'status': 'pending'},
                {'name': 'content_processing', 'status': 'pending'},
                {'name': 'knowledge_extraction', 'status': 'pending'},
                {'name': 'knowledge_base_building', 'status': 'pending'},
                {'name': 'strategy_generation', 'status': 'pending' 
                 if options and options.get('generate_strategies', False) else 'skipped'}
            ],
            'current_step': 'digitization',
            'logs': [{'timestamp': datetime.now(), 'message': 'Job created'}]
        }
        
        result = self.extraction_jobs_collection.insert_one(job)
        job_id = str(result.inserted_id)
        
        # Update with ID
        self.extraction_jobs_collection.update_one(
            {'_id': result.inserted_id},
            {'$set': {'job_id': job_id}}
        )
        
        self.logger.info(f"Created knowledge extraction job with ID: {job_id}")
        
        # Publish event
        self.event_system.publish('knowledge_extraction_job_created', {
            'job_id': job_id,
            'book_path': book_path,
            'timestamp': datetime.now()
        })
        
        return job_id
    
    def _update_job_status(self, job_id: str, status: str, message: str):
        """
        Update the status of a knowledge extraction job.
        
        Args:
            job_id: Job ID
            status: New status
            message: Status message
        """
        # Calculate progress based on status
        progress_map = {
            'created': 0,
            'digitized': 20,
            'content_processed': 40,
            'knowledge_extracted': 60,
            'knowledge_base_built': 80,
            'strategies_generated': 90,
            'completed': 100,
            'failed': -1
        }
        
        progress = progress_map.get(status, 0)
        
        # Update job status
        self.extraction_jobs_collection.update_one(
            {'job_id': job_id},
            {'$set': {
                'status': status,
                'updated_at': datetime.now(),
                'progress': progress,
                'current_step': self._get_current_step(status)
            },
            '$push': {
                'logs': {'timestamp': datetime.now(), 'message': message}
            }}
        )
        
        # Update step status
        step_name = self._get_step_name(status)
        if step_name:
            self.extraction_jobs_collection.update_one(
                {'job_id': job_id, 'steps.name': step_name},
                {'$set': {'steps.$.status': 'completed' if status != 'failed' else 'failed'}}
            )
            
            # Update next step status if not failed
            if status != 'failed':
                next_step = self._get_next_step(step_name)
                if next_step:
                    self.extraction_jobs_collection.update_one(
                        {'job_id': job_id, 'steps.name': next_step},
                        {'$set': {'steps.$.status': 'in_progress'}}
                    )
        
        # Publish event
        self.event_system.publish('knowledge_extraction_job_updated', {
            'job_id': job_id,
            'status': status,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now()
        })
    
    def _get_step_name(self, status: str) -> Optional[str]:
        """
        Get the step name corresponding to a status.
        
        Args:
            status: Status string
            
        Returns:
            Step name or None
        """
        status_to_step = {
            'digitized': 'digitization',
            'content_processed': 'content_processing',
            'knowledge_extracted': 'knowledge_extraction',
            'knowledge_base_built': 'knowledge_base_building',
            'strategies_generated': 'strategy_generation'
        }
        
        return status_to_step.get(status)
    
    def _get_current_step(self, status: str) -> str:
        """
        Get the current step name based on status.
        
        Args:
            status: Status string
            
        Returns:
            Current step name
        """
        if status == 'failed':
            return 'failed'
        elif status == 'completed':
            return 'completed'
        
        step_name = self._get_step_name(status)
        if step_name:
            next_step = self._get_next_step(step_name)
            return next_step if next_step else 'completed'
        
        return 'digitization'  # Default to first step
    
    def _get_next_step(self, step_name: str) -> Optional[str]:
        """
        Get the next step in the pipeline.
        
        Args:
            step_name: Current step name
            
        Returns:
            Next step name or None
        """
        steps = [
            'digitization',
            'content_processing',
            'knowledge_extraction',
            'knowledge_base_building',
            'strategy_generation'
        ]
        
        try:
            current_index = steps.index(step_name)
            if current_index < len(steps) - 1:
                return steps[current_index + 1]
        except ValueError:
            pass
        
        return None
    
    def _digitize_book(self, book_path: str, metadata: Optional[Dict[str, Any]], 
                      job_id: str) -> Dict[str, Any]:
        """
        Digitize a book using OCR.
        
        Args:
            book_path: Path to the book
            metadata: Book metadata
            job_id: Job ID
            
        Returns:
            Dictionary with digitization results
        """
        self.logger.info(f"Digitizing book: {book_path}")
        
        # Check if book is already in digital format
        if book_path.lower().endswith(('.txt', '.pdf', '.epub', '.mobi', '.azw', '.docx')):
            # Book is already in digital format, use OCR digitizer to process it
            result = self.ocr_digitizer.digitize_book(book_path, metadata)
        else:
            # Assume it's a directory of scanned images
            result = self.ocr_digitizer.digitize_book(book_path, metadata)
        
        if result['success']:
            self.logger.info(f"Book digitized successfully with ID: {result['book_id']}")
        else:
            self.logger.error(f"Failed to digitize book: {result.get('error', 'Unknown error')}")
        
        return result
    
    def _process_content(self, book_id: str, job_id: str) -> Dict[str, Any]:
        """
        Process multimodal content from a digitized book.
        
        Args:
            book_id: ID of the digitized book
            job_id: Job ID
            
        Returns:
            Dictionary with content processing results
        """
        self.logger.info(f"Processing content for book ID: {book_id}")
        
        try:
            # Get book pages
            pages_result = self.ocr_digitizer.get_book_pages(book_id)
            if not pages_result['success']:
                return pages_result
            
            pages = pages_result['pages']
            
            # Process each page with the multimodal content processor
            content_ids = []
            for page in pages:
                # Determine content type based on page content
                if 'content' in page:
                    content = page['content']
                    
                    # Process text content
                    if 'text' in content and content['text']:
                        text_result = self.content_processor.process_text_content(
                            content['text'], 
                            {'page_id': page.get('page_id'), 'page_number': page.get('page_number')}
                        )
                        if text_result['success']:
                            content_ids.append(text_result['content_id'])
                    
                    # Process tables
                    if 'tables' in content and content['tables']:
                        for table in content['tables']:
                            table_result = self.content_processor.process_table_content(
                                table, 
                                {'page_id': page.get('page_id'), 'page_number': page.get('page_number')}
                            )
                            if table_result['success']:
                                content_ids.append(table_result['content_id'])
                    
                    # Process images
                    if 'images' in content and content['images']:
                        for image in content['images']:
                            image_result = self.content_processor.process_image_content(
                                image, 
                                {'page_id': page.get('page_id'), 'page_number': page.get('page_number')}
                            )
                            if image_result['success']:
                                content_ids.append(image_result['content_id'])
                    
                    # Process formulas
                    if 'formulas' in content and content['formulas']:
                        for formula in content['formulas']:
                            formula_result = self.content_processor.process_math_content(
                                formula, 
                                {'page_id': page.get('page_id'), 'page_number': page.get('page_number')}
                            )
                            if formula_result['success']:
                                content_ids.append(formula_result['content_id'])
            
            return {
                'success': True,
                'book_id': book_id,
                'content_ids': content_ids,
                'content_count': len(content_ids)
            }
        
        except Exception as e:
            error_msg = f"Error processing content: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _extract_knowledge(self, book_id: str, content_ids: List[str], 
                         job_id: str) -> Dict[str, Any]:
        """
        Extract knowledge from processed content.
        
        Args:
            book_id: ID of the digitized book
            content_ids: IDs of processed content items
            job_id: Job ID
            
        Returns:
            Dictionary with knowledge extraction results
        """
        self.logger.info(f"Extracting knowledge from book ID: {book_id}")
        
        try:
            # Get book metadata
            book_result = self.ocr_digitizer.get_digitized_book(book_id)
            if not book_result['success']:
                return book_result
            
            book = book_result['book']
            
            # Get processed content items
            content_items = []
            for content_id in content_ids:
                content_result = self.content_processor.get_processed_content(content_id)
                if content_result['success']:
                    content_items.append(content_result['content'])
            
            # Extract knowledge using the book knowledge extractor
            extraction_result = self.knowledge_extractor.extract_knowledge_from_content(
                content_items,
                {
                    'book_id': book_id,
                    'book_title': book.get('title', 'Unknown'),
                    'book_author': book.get('author', 'Unknown'),
                    'book_publisher': book.get('publisher', 'Unknown'),
                    'book_year': book.get('publication_year', 'Unknown')
                }
            )
            
            if not extraction_result['success']:
                return extraction_result
            
            return {
                'success': True,
                'book_id': book_id,
                'knowledge_items': extraction_result['knowledge_items'],
                'knowledge_count': len(extraction_result['knowledge_items'])
            }
        
        except Exception as e:
            error_msg = f"Error extracting knowledge: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _build_knowledge_base(self, knowledge_items: List[Dict[str, Any]], 
                            book_id: str, job_id: str) -> Dict[str, Any]:
        """
        Add extracted knowledge to the knowledge base.
        
        Args:
            knowledge_items: List of extracted knowledge items
            book_id: ID of the digitized book
            job_id: Job ID
            
        Returns:
            Dictionary with knowledge base building results
        """
        self.logger.info(f"Building knowledge base with {len(knowledge_items)} items from book ID: {book_id}")
        
        try:
            # Add knowledge items to the knowledge base
            kb_result = self.knowledge_base.add_knowledge_items(knowledge_items)
            
            if not kb_result['success']:
                return kb_result
            
            # Update knowledge base statistics
            self.knowledge_base.update_statistics()
            
            return {
                'success': True,
                'book_id': book_id,
                'knowledge_base_ids': kb_result['knowledge_ids'],
                'knowledge_count': len(kb_result['knowledge_ids'])
            }
        
        except Exception as e:
            error_msg = f"Error building knowledge base: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _generate_strategies(self, knowledge_ids: List[str], job_id: str) -> Dict[str, Any]:
        """
        Generate trading strategies from knowledge base items.
        
        Args:
            knowledge_ids: IDs of knowledge base items
            job_id: Job ID
            
        Returns:
            Dictionary with strategy generation results
        """
        self.logger.info(f"Generating strategies from {len(knowledge_ids)} knowledge items")
        
        try:
            # Get knowledge items from the knowledge base
            knowledge_items = []
            for kb_id in knowledge_ids:
                kb_result = self.knowledge_base.get_knowledge_item(kb_id)
                if kb_result['success']:
                    knowledge_items.append(kb_result['knowledge_item'])
            
            # Generate strategies
            strategy_result = self.strategy_generator.generate_strategies_batch(knowledge_items)
            
            if not strategy_result['success']:
                return strategy_result
            
            return {
                'success': True,
                'strategy_ids': strategy_result['strategy_ids'],
                'strategy_count': len(strategy_result['strategy_ids'])
            }
        
        except Exception as e:
            error_msg = f"Error generating strategies: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get the status of a knowledge extraction job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dictionary with job status
        """
        try:
            job = self.extraction_jobs_collection.find_one({'job_id': job_id})
            if not job:
                return {'success': False, 'error': 'Job not found'}
            
            # Convert ObjectId to string for JSON serialization
            job['_id'] = str(job['_id'])
            
            return {'success': True, 'job': job}
        
        except Exception as e:
            error_msg = f"Error getting job status: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def list_jobs(self, status: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
        """
        List knowledge extraction jobs.
        
        Args:
            status: Optional status filter
            limit: Maximum number of jobs to return
            
        Returns:
            Dictionary with job list
        """
        try:
            # Build query
            query = {}
            if status:
                query['status'] = status
            
            # Get jobs
            jobs = list(self.extraction_jobs_collection.find(query).sort('created_at', -1).limit(limit))
            
            # Convert ObjectId to string for JSON serialization
            for job in jobs:
                job['_id'] = str(job['_id'])
            
            return {'success': True, 'jobs': jobs, 'count': len(jobs)}
        
        except Exception as e:
            error_msg = f"Error listing jobs: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """
        Cancel a knowledge extraction job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dictionary with cancellation result
        """
        try:
            # Check if job exists
            job = self.extraction_jobs_collection.find_one({'job_id': job_id})
            if not job:
                return {'success': False, 'error': 'Job not found'}
            
            # Check if job can be cancelled
            if job['status'] in ['completed', 'failed', 'cancelled']:
                return {'success': False, 'error': f"Job cannot be cancelled in {job['status']} status"}
            
            # Update job status
            self._update_job_status(job_id, 'cancelled', "Job cancelled by user")
            
            return {'success': True, 'job_id': job_id}
        
        except Exception as e:
            error_msg = f"Error cancelling job: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def retry_job(self, job_id: str) -> Dict[str, Any]:
        """
        Retry a failed knowledge extraction job.
        
        Args:
            job_id: Job ID
            
        Returns:
            Dictionary with retry result
        """
        try:
            # Check if job exists
            job_result = self.get_job_status(job_id)
            if not job_result['success']:
                return job_result
            
            job = job_result['job']
            
            # Check if job can be retried
            if job['status'] not in ['failed', 'cancelled']:
                return {'success': False, 'error': f"Job cannot be retried in {job['status']} status"}
            
            # Create a new job with the same parameters
            new_job_id = self._create_extraction_job(
                job['book_path'],
                job.get('metadata', {}),
                job.get('options', {})
            )
            
            # Update old job status
            self._update_job_status(job_id, 'retried', f"Job retried with new job ID: {new_job_id}")
            
            # Start processing the new job
            result = self.process_book(
                job['book_path'],
                job.get('metadata', {}),
                job.get('options', {})
            )
            
            return {'success': True, 'old_job_id': job_id, 'new_job_id': new_job_id, 'result': result}
        
        except Exception as e:
            error_msg = f"Error retrying job: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def batch_process_books(self, book_paths: List[str], 
                          metadata: Optional[Dict[str, Any]] = None,
                          options: Optional[Dict[str, Any]] = None,
                          parallel: bool = False) -> Dict[str, Any]:
        """
        Process multiple books in batch mode.
        
        Args:
            book_paths: List of paths to books to process
            metadata: Optional metadata to apply to all books
            options: Optional processing options
            parallel: Whether to process books in parallel
            
        Returns:
            Dictionary with batch processing results
        """
        self.logger.info(f"Starting batch processing of {len(book_paths)} books")
        
        results = {
            'total': len(book_paths),
            'successful': 0,
            'failed': 0,
            'job_ids': [],
            'errors': []
        }
        
        if parallel:
            # Process books in parallel using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_path = {executor.submit(self.process_book, path, metadata, options): path 
                                 for path in book_paths}
                
                for future in concurrent.futures.as_completed(future_to_path):
                    path = future_to_path[future]
                    try:
                        result = future.result()
                        self._update_batch_results(results, path, result)
                    except Exception as e:
                        results['failed'] += 1
                        results['errors'].append({
                            'path': path,
                            'error': str(e)
                        })
        else:
            # Process books sequentially
            for path in book_paths:
                result = self.process_book(path, metadata, options)
                self._update_batch_results(results, path, result)
        
        # Publish batch event
        self.event_system.publish('knowledge_extraction_batch_completed', {
            'total': results['total'],
            'successful': results['successful'],
            'failed': results['failed'],
            'timestamp': datetime.now()
        })
        
        return results
    
    def _update_batch_results(self, results: Dict[str, Any], path: str, result: Dict[str, Any]):
        """
        Update batch processing results.
        
        Args:
            results: Results dictionary to update
            path: Book path
            result: Processing result
        """
        if result.get('success', False):
            results['successful'] += 1
            results['job_ids'].append(result.get('job_id'))
        else:
            results['failed'] += 1
            results['errors'].append({
                'path': path,
                'error': result.get('error', 'Unknown error'),
                'job_id': result.get('job_id')
            })
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the Knowledge Extraction System.
        
        Returns:
            Dictionary with system statistics
        """
        try:
            # Get job statistics
            total_jobs = self.extraction_jobs_collection.count_documents({})
            completed_jobs = self.extraction_jobs_collection.count_documents({'status': 'completed'})
            failed_jobs = self.extraction_jobs_collection.count_documents({'status': 'failed'})
            in_progress_jobs = self.extraction_jobs_collection.count_documents(
                {'status': {'$nin': ['completed', 'failed', 'cancelled']}})
            
            # Get knowledge base statistics
            kb_stats = self.knowledge_base.get_statistics()
            
            # Get strategy statistics
            strategy_count = self.strategy_generator.count_strategies()
            
            return {
                'success': True,
                'jobs': {
                    'total': total_jobs,
                    'completed': completed_jobs,
                    'failed': failed_jobs,
                    'in_progress': in_progress_jobs,
                    'success_rate': (completed_jobs / total_jobs * 100) if total_jobs > 0 else 0
                },
                'knowledge_base': kb_stats.get('statistics', {}),
                'strategies': {
                    'total': strategy_count
                },
                'timestamp': datetime.now()
            }
        
        except Exception as e:
            error_msg = f"Error getting system statistics: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def export_knowledge_to_json(self, output_path: str) -> Dict[str, Any]:
        """
        Export the entire knowledge base to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            Dictionary with export result
        """
        return self.knowledge_base.export_to_json(output_path)
    
    def import_knowledge_from_json(self, input_path: str) -> Dict[str, Any]:
        """
        Import knowledge from a JSON file into the knowledge base.
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            Dictionary with import result
        """
        return self.knowledge_base.import_from_json(input_path)
    
    def export_strategies_to_json(self, output_path: str) -> Dict[str, Any]:
        """
        Export all strategies to a JSON file.
        
        Args:
            output_path: Path to save the JSON file
            
        Returns:
            Dictionary with export result
        """
        return self.strategy_generator.export_strategies(output_path)
    
    def import_strategies_from_json(self, input_path: str) -> Dict[str, Any]:
        """
        Import strategies from a JSON file.
        
        Args:
            input_path: Path to the JSON file
            
        Returns:
            Dictionary with import result
        """
        return self.strategy_generator.import_strategies(input_path)
    
    def search_knowledge_base(self, query: str, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search the knowledge base.
        
        Args:
            query: Search query
            filters: Optional filters
            
        Returns:
            Dictionary with search results
        """
        return self.knowledge_base.search(query, filters)
    
    def get_strategies_for_knowledge(self, knowledge_id: str) -> Dict[str, Any]:
        """
        Get strategies generated from a specific knowledge item.
        
        Args:
            knowledge_id: Knowledge item ID
            
        Returns:
            Dictionary with strategies
        """
        try:
            # Get the knowledge item
            kb_result = self.knowledge_base.get_knowledge_item(knowledge_id)
            if not kb_result['success']:
                return kb_result
            
            knowledge_item = kb_result['knowledge_item']
            
            # Get strategies that reference this knowledge item
            strategies = self.strategy_generator.list_strategies(
                {'knowledge_source_id': knowledge_id}
            )
            
            return {
                'success': True,
                'knowledge_id': knowledge_id,
                'knowledge_title': knowledge_item.get('title', 'Unknown'),
                'strategies': strategies.get('strategies', []),
                'count': len(strategies.get('strategies', []))
            }
        
        except Exception as e:
            error_msg = f"Error getting strategies for knowledge: {str(e)}"
            self.logger.error(error_msg)
            return {'success': False, 'error': error_msg}