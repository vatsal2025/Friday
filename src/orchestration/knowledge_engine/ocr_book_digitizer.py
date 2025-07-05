import os
import logging
import json
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import base64

# Import necessary infrastructure components
from src.infrastructure.config.config_manager import ConfigManager
from src.infrastructure.database import mongodb
from src.infrastructure.event.event_system import EventSystem
from src.infrastructure.logging import get_logger

class OCRBookDigitizer:
    """
    Handles the digitization of physical books using OCR technology.
    
    This component is responsible for converting physical books to digital format,
    extracting text, tables, and images from scanned book pages, and preparing
    the content for further processing by the Knowledge Extraction System.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the OCRBookDigitizer with configuration settings.
        
        Args:
            config: Optional configuration dictionary. If None, loads from unified_config.
        """
        self.config_manager = ConfigManager()
        self.config = config if config else self.config_manager.get_config('knowledge_extraction')
        
        # Initialize database connection
        self.digitized_books_collection = mongodb.get_collection('digitized_books')
        self.book_pages_collection = mongodb.get_collection('book_pages')
        
        # Initialize event system
        self.event_system = EventSystem()
        
        # Configure logging
        self.logger = get_logger(__name__)
        
        # Load OCR configurations
        self._load_ocr_config()
    
    def _load_ocr_config(self):
        """
        Load OCR-specific configurations from the main config.
        """
        # OCR engine config
        self.ocr_config = self.config.get('ocr', {})
        self.ocr_engine = self.ocr_config.get('engine', 'tesseract')
        self.ocr_language = self.ocr_config.get('language', 'eng')
        self.ocr_dpi = self.ocr_config.get('dpi', 300)
        self.ocr_confidence_threshold = self.ocr_config.get('confidence_threshold', 0.7)
        
        # Image preprocessing config
        self.preprocessing_config = self.ocr_config.get('preprocessing', {})
        self.enable_preprocessing = self.preprocessing_config.get('enabled', True)
        self.preprocessing_methods = self.preprocessing_config.get('methods', ['grayscale', 'denoise', 'deskew'])
        
        # Layout analysis config
        self.layout_config = self.ocr_config.get('layout_analysis', {})
        self.enable_layout_analysis = self.layout_config.get('enabled', True)
        self.detect_tables = self.layout_config.get('detect_tables', True)
        self.detect_images = self.layout_config.get('detect_images', True)
        self.detect_formulas = self.layout_config.get('detect_formulas', True)
        
        # Output format config
        self.output_config = self.ocr_config.get('output', {})
        self.output_formats = self.output_config.get('formats', ['text', 'json', 'pdf'])
        self.save_intermediate_results = self.output_config.get('save_intermediate_results', False)
        
        self.logger.info(f"OCR Book Digitizer configured with engine: {self.ocr_engine}")
    
    def digitize_book(self, book_path: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Digitize a book from a PDF or a directory of scanned images.
        
        Args:
            book_path: Path to the book PDF or directory of scanned images
            metadata: Optional metadata about the book
            
        Returns:
            Dictionary with digitization results
        """
        self.logger.info(f"Starting book digitization: {book_path}")
        
        if not os.path.exists(book_path):
            self.logger.error(f"Book path not found: {book_path}")
            return {'success': False, 'error': 'Book path not found'}
        
        try:
            # Initialize book metadata
            book_metadata = metadata or {}
            book_metadata.update({
                'digitized_at': datetime.now(),
                'source_path': book_path,
                'ocr_engine': self.ocr_engine,
                'ocr_language': self.ocr_language,
                'ocr_dpi': self.ocr_dpi,
                'preprocessing_applied': self.enable_preprocessing,
                'preprocessing_methods': self.preprocessing_methods if self.enable_preprocessing else []
            })
            
            # Determine if input is a PDF or directory
            is_pdf = book_path.lower().endswith('.pdf')
            
            # Process the book
            if is_pdf:
                result = self._process_pdf_book(book_path, book_metadata)
            else:
                result = self._process_image_directory(book_path, book_metadata)
            
            if result['success']:
                # Save book metadata to database
                book_id = self._save_digitized_book(result['book_data'])
                result['book_id'] = book_id
                
                # Publish event
                self.event_system.publish('book_digitized', {
                    'book_id': book_id,
                    'title': result['book_data'].get('title', 'Unknown'),
                    'pages_count': result['book_data'].get('pages_count', 0),
                    'timestamp': datetime.now()
                })
            
            return result
        
        except Exception as e:
            self.logger.error(f"Error digitizing book: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _process_pdf_book(self, pdf_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a book in PDF format.
        
        Args:
            pdf_path: Path to the PDF file
            metadata: Book metadata
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing PDF book: {pdf_path}")
        
        # In a production system, this would use PDF processing libraries
        # like PyPDF2, pdf2image, or PyMuPDF to extract pages and perform OCR
        
        # For this implementation, we'll create a simplified result
        book_data = {
            'title': metadata.get('title', os.path.basename(pdf_path)),
            'author': metadata.get('author', 'Unknown'),
            'publisher': metadata.get('publisher', 'Unknown'),
            'publication_year': metadata.get('publication_year', 'Unknown'),
            'isbn': metadata.get('isbn', 'Unknown'),
            'language': metadata.get('language', 'en'),
            'pages_count': 100,  # Example value
            'file_size_bytes': os.path.getsize(pdf_path),
            'file_type': 'pdf',
            'metadata': metadata,
            'pages': []
        }
        
        # Simulate processing pages
        for page_num in range(1, 11):  # Process first 10 pages as example
            page_data = self._simulate_page_processing(page_num, 'pdf')
            book_data['pages'].append(page_data)
            
            # Save page to database
            page_id = self._save_book_page(page_data, book_data['title'], page_num)
            page_data['page_id'] = page_id
        
        return {'success': True, 'book_data': book_data}
    
    def _process_image_directory(self, dir_path: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a directory of book page images.
        
        Args:
            dir_path: Path to the directory containing page images
            metadata: Book metadata
            
        Returns:
            Dictionary with processing results
        """
        self.logger.info(f"Processing book image directory: {dir_path}")
        
        # Get image files in the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp']
        image_files = []
        
        for file in os.listdir(dir_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(dir_path, file))
        
        image_files.sort()  # Sort to maintain page order
        
        if not image_files:
            return {'success': False, 'error': 'No image files found in directory'}
        
        # Create book data
        book_data = {
            'title': metadata.get('title', os.path.basename(dir_path)),
            'author': metadata.get('author', 'Unknown'),
            'publisher': metadata.get('publisher', 'Unknown'),
            'publication_year': metadata.get('publication_year', 'Unknown'),
            'isbn': metadata.get('isbn', 'Unknown'),
            'language': metadata.get('language', 'en'),
            'pages_count': len(image_files),
            'file_type': 'image_directory',
            'metadata': metadata,
            'pages': []
        }
        
        # Process each image as a page
        for page_num, image_path in enumerate(image_files, 1):
            # In a production system, this would use actual OCR processing
            # For this implementation, we'll simulate the processing
            page_data = self._simulate_page_processing(page_num, 'image', image_path)
            book_data['pages'].append(page_data)
            
            # Save page to database
            page_id = self._save_book_page(page_data, book_data['title'], page_num)
            page_data['page_id'] = page_id
        
        return {'success': True, 'book_data': book_data}
    
    def _simulate_page_processing(self, page_num: int, source_type: str, 
                                file_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Simulate OCR processing of a page for demonstration purposes.
        
        Args:
            page_num: Page number
            source_type: Source type ('pdf' or 'image')
            file_path: Optional path to the image file
            
        Returns:
            Dictionary with simulated page data
        """
        # In a production system, this would perform actual OCR and layout analysis
        
        # Create simulated page data
        page_data = {
            'page_number': page_num,
            'source_type': source_type,
            'source_path': file_path,
            'processed_at': datetime.now(),
            'ocr_confidence': 0.85,  # Example value
            'content': {
                'text': f"This is simulated text content for page {page_num}. "
                       f"It would contain the actual OCR-extracted text in a production system.",
                'tables': [],
                'images': [],
                'formulas': []
            },
            'layout': {
                'paragraphs': [
                    {
                        'text': f"Paragraph 1 on page {page_num}",
                        'bbox': [100, 100, 500, 150],  # Example bounding box [x1, y1, x2, y2]
                        'confidence': 0.9
                    },
                    {
                        'text': f"Paragraph 2 on page {page_num}",
                        'bbox': [100, 200, 500, 250],
                        'confidence': 0.85
                    }
                ],
                'page_dimensions': {'width': 612, 'height': 792}  # Standard letter size
            }
        }
        
        # Add simulated tables if enabled
        if self.detect_tables and page_num % 3 == 0:  # Add a table to every third page
            page_data['content']['tables'].append({
                'rows': 3,
                'columns': 3,
                'bbox': [100, 300, 500, 400],
                'confidence': 0.8,
                'data': [
                    ['Header 1', 'Header 2', 'Header 3'],
                    ['Value 1', 'Value 2', 'Value 3'],
                    ['Value 4', 'Value 5', 'Value 6']
                ]
            })
        
        # Add simulated images if enabled
        if self.detect_images and page_num % 4 == 0:  # Add an image to every fourth page
            page_data['content']['images'].append({
                'bbox': [150, 450, 450, 550],
                'confidence': 0.9,
                'caption': f"Figure {page_num//4}",
                'image_type': 'chart'  # or 'photograph', 'diagram', etc.
            })
        
        # Add simulated formulas if enabled
        if self.detect_formulas and page_num % 5 == 0:  # Add a formula to every fifth page
            page_data['content']['formulas'].append({
                'bbox': [200, 600, 400, 650],
                'confidence': 0.75,
                'latex': 'E = mc^2',
                'variables': ['E', 'm', 'c']
            })
        
        return page_data
    
    def _save_digitized_book(self, book_data: Dict[str, Any]) -> str:
        """
        Save digitized book data to the database.
        
        Args:
            book_data: Book data to save
            
        Returns:
            ID of the saved book
        """
        # Remove large page content to save space in the main book document
        book_data_to_save = book_data.copy()
        book_data_to_save['pages'] = [{'page_number': p['page_number'], 'page_id': p.get('page_id')} 
                                     for p in book_data['pages']]
        
        # Insert into database
        result = self.digitized_books_collection.insert_one(book_data_to_save)
        book_id = str(result.inserted_id)
        
        # Update with ID
        self.digitized_books_collection.update_one(
            {'_id': result.inserted_id},
            {'$set': {'book_id': book_id}}
        )
        
        self.logger.info(f"Saved digitized book with ID: {book_id}")
        
        return book_id
    
    def _save_book_page(self, page_data: Dict[str, Any], book_title: str, page_number: int) -> str:
        """
        Save a book page to the database.
        
        Args:
            page_data: Page data to save
            book_title: Title of the book
            page_number: Page number
            
        Returns:
            ID of the saved page
        """
        # Add book reference
        page_data['book_title'] = book_title
        page_data['page_number'] = page_number
        
        # Insert into database
        result = self.book_pages_collection.insert_one(page_data)
        page_id = str(result.inserted_id)
        
        # Update with ID
        self.book_pages_collection.update_one(
            {'_id': result.inserted_id},
            {'$set': {'page_id': page_id}}
        )
        
        return page_id
    
    def get_digitized_book(self, book_id: str) -> Dict[str, Any]:
        """
        Get a digitized book by ID.
        
        Args:
            book_id: ID of the book to get
            
        Returns:
            Dictionary with the book data or error
        """
        try:
            book = self.digitized_books_collection.find_one({'book_id': book_id})
            if not book:
                return {'success': False, 'error': 'Book not found'}
            
            # Convert ObjectId to string for JSON serialization
            book['_id'] = str(book['_id'])
            
            return {'success': True, 'book': book}
        
        except Exception as e:
            self.logger.error(f"Error getting digitized book: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_book_page(self, page_id: str) -> Dict[str, Any]:
        """
        Get a book page by ID.
        
        Args:
            page_id: ID of the page to get
            
        Returns:
            Dictionary with the page data or error
        """
        try:
            page = self.book_pages_collection.find_one({'page_id': page_id})
            if not page:
                return {'success': False, 'error': 'Page not found'}
            
            # Convert ObjectId to string for JSON serialization
            page['_id'] = str(page['_id'])
            
            return {'success': True, 'page': page}
        
        except Exception as e:
            self.logger.error(f"Error getting book page: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_book_pages(self, book_id: str, page_numbers: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Get pages from a digitized book.
        
        Args:
            book_id: ID of the book
            page_numbers: Optional list of page numbers to get. If None, gets all pages.
            
        Returns:
            Dictionary with the pages data or error
        """
        try:
            # Get the book first to verify it exists and get page IDs
            book_result = self.get_digitized_book(book_id)
            if not book_result['success']:
                return book_result
            
            book = book_result['book']
            
            # Filter pages if page_numbers is provided
            if page_numbers:
                page_ids = [p['page_id'] for p in book['pages'] 
                           if p['page_number'] in page_numbers]
            else:
                page_ids = [p['page_id'] for p in book['pages']]
            
            if not page_ids:
                return {'success': False, 'error': 'No matching pages found'}
            
            # Get the pages
            pages = []
            for page_id in page_ids:
                page_result = self.get_book_page(page_id)
                if page_result['success']:
                    pages.append(page_result['page'])
            
            return {'success': True, 'pages': pages, 'count': len(pages)}
        
        except Exception as e:
            self.logger.error(f"Error getting book pages: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def extract_book_text(self, book_id: str, include_tables: bool = True) -> Dict[str, Any]:
        """
        Extract all text content from a digitized book.
        
        Args:
            book_id: ID of the book
            include_tables: Whether to include text from tables
            
        Returns:
            Dictionary with the extracted text or error
        """
        try:
            # Get all pages
            pages_result = self.get_book_pages(book_id)
            if not pages_result['success']:
                return pages_result
            
            pages = pages_result['pages']
            
            # Extract text from each page
            full_text = ""
            for page in sorted(pages, key=lambda p: p['page_number']):
                # Add page text
                if 'content' in page and 'text' in page['content']:
                    full_text += f"\n\n--- Page {page['page_number']} ---\n\n"
                    full_text += page['content']['text']
                
                # Add table text if requested
                if include_tables and 'content' in page and 'tables' in page['content']:
                    for table in page['content']['tables']:
                        full_text += f"\n\n--- Table on Page {page['page_number']} ---\n\n"
                        if 'data' in table:
                            for row in table['data']:
                                full_text += ' | '.join(str(cell) for cell in row) + '\n'
            
            return {
                'success': True, 
                'book_id': book_id,
                'text': full_text,
                'pages_count': len(pages)
            }
        
        except Exception as e:
            self.logger.error(f"Error extracting book text: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def extract_book_tables(self, book_id: str) -> Dict[str, Any]:
        """
        Extract all tables from a digitized book.
        
        Args:
            book_id: ID of the book
            
        Returns:
            Dictionary with the extracted tables or error
        """
        try:
            # Get all pages
            pages_result = self.get_book_pages(book_id)
            if not pages_result['success']:
                return pages_result
            
            pages = pages_result['pages']
            
            # Extract tables from each page
            tables = []
            for page in sorted(pages, key=lambda p: p['page_number']):
                if 'content' in page and 'tables' in page['content']:
                    for table in page['content']['tables']:
                        table_data = table.copy()
                        table_data['page_number'] = page['page_number']
                        tables.append(table_data)
            
            return {
                'success': True, 
                'book_id': book_id,
                'tables': tables,
                'tables_count': len(tables)
            }
        
        except Exception as e:
            self.logger.error(f"Error extracting book tables: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def extract_book_images(self, book_id: str) -> Dict[str, Any]:
        """
        Extract all images from a digitized book.
        
        Args:
            book_id: ID of the book
            
        Returns:
            Dictionary with the extracted images or error
        """
        try:
            # Get all pages
            pages_result = self.get_book_pages(book_id)
            if not pages_result['success']:
                return pages_result
            
            pages = pages_result['pages']
            
            # Extract images from each page
            images = []
            for page in sorted(pages, key=lambda p: p['page_number']):
                if 'content' in page and 'images' in page['content']:
                    for image in page['content']['images']:
                        image_data = image.copy()
                        image_data['page_number'] = page['page_number']
                        images.append(image_data)
            
            return {
                'success': True, 
                'book_id': book_id,
                'images': images,
                'images_count': len(images)
            }
        
        except Exception as e:
            self.logger.error(f"Error extracting book images: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def extract_book_formulas(self, book_id: str) -> Dict[str, Any]:
        """
        Extract all mathematical formulas from a digitized book.
        
        Args:
            book_id: ID of the book
            
        Returns:
            Dictionary with the extracted formulas or error
        """
        try:
            # Get all pages
            pages_result = self.get_book_pages(book_id)
            if not pages_result['success']:
                return pages_result
            
            pages = pages_result['pages']
            
            # Extract formulas from each page
            formulas = []
            for page in sorted(pages, key=lambda p: p['page_number']):
                if 'content' in page and 'formulas' in page['content']:
                    for formula in page['content']['formulas']:
                        formula_data = formula.copy()
                        formula_data['page_number'] = page['page_number']
                        formulas.append(formula_data)
            
            return {
                'success': True, 
                'book_id': book_id,
                'formulas': formulas,
                'formulas_count': len(formulas)
            }
        
        except Exception as e:
            self.logger.error(f"Error extracting book formulas: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def export_book(self, book_id: str, export_format: str = 'json', 
                  output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Export a digitized book to a specified format.
        
        Args:
            book_id: ID of the book to export
            export_format: Format to export to ('json', 'text', 'pdf', 'html')
            output_path: Path to save the exported file. If None, returns content directly.
            
        Returns:
            Dictionary with export results
        """
        try:
            # Get the book
            book_result = self.get_digitized_book(book_id)
            if not book_result['success']:
                return book_result
            
            book = book_result['book']
            
            # Get all pages
            pages_result = self.get_book_pages(book_id)
            if not pages_result['success']:
                return pages_result
            
            pages = pages_result['pages']
            
            # Export based on format
            if export_format == 'json':
                # Create JSON export
                export_data = {
                    'book': book,
                    'pages': pages,
                    'exported_at': datetime.now().isoformat(),
                    'export_format': 'json'
                }
                
                if output_path:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
                    return {'success': True, 'file_path': output_path}
                else:
                    return {'success': True, 'content': export_data}
            
            elif export_format == 'text':
                # Extract all text
                text_result = self.extract_book_text(book_id, include_tables=True)
                if not text_result['success']:
                    return text_result
                
                if output_path:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text_result['text'])
                    return {'success': True, 'file_path': output_path}
                else:
                    return {'success': True, 'content': text_result['text']}
            
            elif export_format in ['pdf', 'html']:
                # These would require additional libraries in a production system
                return {'success': False, 'error': f"Export to {export_format} not implemented in this version"}
            
            else:
                return {'success': False, 'error': f"Unsupported export format: {export_format}"}
        
        except Exception as e:
            self.logger.error(f"Error exporting book: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def search_book_content(self, book_id: str, query: str) -> Dict[str, Any]:
        """
        Search for content within a digitized book.
        
        Args:
            book_id: ID of the book to search
            query: Search query
            
        Returns:
            Dictionary with search results
        """
        try:
            # Get all pages
            pages_result = self.get_book_pages(book_id)
            if not pages_result['success']:
                return pages_result
            
            pages = pages_result['pages']
            
            # Search in page content
            results = []
            for page in pages:
                # Search in main text
                if 'content' in page and 'text' in page['content']:
                    text = page['content']['text']
                    if query.lower() in text.lower():
                        # Find the context around the match
                        index = text.lower().find(query.lower())
                        start = max(0, index - 50)
                        end = min(len(text), index + len(query) + 50)
                        context = text[start:end]
                        
                        results.append({
                            'page_number': page['page_number'],
                            'content_type': 'text',
                            'context': context,
                            'match_position': index
                        })
                
                # Search in tables
                if 'content' in page and 'tables' in page['content']:
                    for table_index, table in enumerate(page['content']['tables']):
                        if 'data' in table:
                            for row_index, row in enumerate(table['data']):
                                for cell_index, cell in enumerate(row):
                                    if query.lower() in str(cell).lower():
                                        results.append({
                                            'page_number': page['page_number'],
                                            'content_type': 'table',
                                            'table_index': table_index,
                                            'row': row_index,
                                            'column': cell_index,
                                            'cell_content': cell
                                        })
            
            return {
                'success': True, 
                'book_id': book_id,
                'query': query,
                'results': results,
                'results_count': len(results)
            }
        
        except Exception as e:
            self.logger.error(f"Error searching book content: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def batch_digitize_books(self, book_paths: List[str], 
                           metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Digitize multiple books in batch mode.
        
        Args:
            book_paths: List of paths to books to digitize
            metadata: Optional metadata to apply to all books
            
        Returns:
            Dictionary with batch digitization results
        """
        results = {
            'total': len(book_paths),
            'successful': 0,
            'failed': 0,
            'book_ids': [],
            'errors': []
        }
        
        for path in book_paths:
            result = self.digitize_book(path, metadata)
            
            if result.get('success', False):
                results['successful'] += 1
                results['book_ids'].append(result.get('book_id'))
            else:
                results['failed'] += 1
                results['errors'].append({
                    'path': path,
                    'error': result.get('error', 'Unknown error')
                })
        
        # Publish batch event
        self.event_system.publish('books_batch_digitized', {
            'total': results['total'],
            'successful': results['successful'],
            'timestamp': datetime.now()
        })
        
        return results
    
    def improve_ocr_quality(self, page_id: str, 
                          preprocessing_methods: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Improve OCR quality for a specific page by applying additional preprocessing.
        
        Args:
            page_id: ID of the page to improve
            preprocessing_methods: Optional list of preprocessing methods to apply
            
        Returns:
            Dictionary with improvement results
        """
        try:
            # Get the page
            page_result = self.get_book_page(page_id)
            if not page_result['success']:
                return page_result
            
            page = page_result['page']
            
            # In a production system, this would apply actual image preprocessing
            # and re-run OCR with different settings
            
            # For this implementation, we'll simulate improvement
            methods = preprocessing_methods or ['contrast_enhancement', 'super_resolution', 'noise_reduction']
            
            # Update page data with "improved" content
            improved_page = page.copy()
            improved_page['ocr_confidence'] = min(0.95, page.get('ocr_confidence', 0.7) + 0.1)
            improved_page['preprocessing_methods'] = methods
            improved_page['improved_at'] = datetime.now()
            
            # Update in database
            self.book_pages_collection.update_one(
                {'page_id': page_id},
                {'$set': {
                    'ocr_confidence': improved_page['ocr_confidence'],
                    'preprocessing_methods': methods,
                    'improved_at': improved_page['improved_at']
                }}
            )
            
            return {
                'success': True, 
                'page_id': page_id,
                'original_confidence': page.get('ocr_confidence', 0.7),
                'improved_confidence': improved_page['ocr_confidence'],
                'methods_applied': methods
            }
        
        except Exception as e:
            self.logger.error(f"Error improving OCR quality: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_book_statistics(self, book_id: str) -> Dict[str, Any]:
        """
        Get statistics about a digitized book.
        
        Args:
            book_id: ID of the book
            
        Returns:
            Dictionary with book statistics
        """
        try:
            # Get the book
            book_result = self.get_digitized_book(book_id)
            if not book_result['success']:
                return book_result
            
            book = book_result['book']
            
            # Get all pages
            pages_result = self.get_book_pages(book_id)
            if not pages_result['success']:
                return pages_result
            
            pages = pages_result['pages']
            
            # Calculate statistics
            total_words = 0
            total_tables = 0
            total_images = 0
            total_formulas = 0
            avg_confidence = 0
            
            for page in pages:
                # Count words
                if 'content' in page and 'text' in page['content']:
                    total_words += len(page['content']['text'].split())
                
                # Count tables
                if 'content' in page and 'tables' in page['content']:
                    total_tables += len(page['content']['tables'])
                
                # Count images
                if 'content' in page and 'images' in page['content']:
                    total_images += len(page['content']['images'])
                
                # Count formulas
                if 'content' in page and 'formulas' in page['content']:
                    total_formulas += len(page['content']['formulas'])
                
                # Sum confidence
                avg_confidence += page.get('ocr_confidence', 0)
            
            # Calculate average confidence
            avg_confidence = avg_confidence / len(pages) if pages else 0
            
            statistics = {
                'book_id': book_id,
                'title': book.get('title', 'Unknown'),
                'pages_count': len(pages),
                'total_words': total_words,
                'total_tables': total_tables,
                'total_images': total_images,
                'total_formulas': total_formulas,
                'avg_ocr_confidence': avg_confidence,
                'words_per_page': total_words / len(pages) if pages else 0
            }
            
            return {'success': True, 'statistics': statistics}
        
        except Exception as e:
            self.logger.error(f"Error getting book statistics: {str(e)}")
            return {'success': False, 'error': str(e)}