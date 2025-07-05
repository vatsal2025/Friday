"""Knowledge Engine API Module.

This module provides REST API endpoints for interacting with the knowledge engine.
"""

import os
import logging
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends, Query, Body
from pydantic import BaseModel, Field

from src.infrastructure.auth.api_key_auth import api_key_auth
from src.orchestration.knowledge_engine.knowledge_engine_service import KnowledgeEngineService


# Define API models
class KnowledgeItem(BaseModel):
    """Model for a knowledge item."""
    id: Optional[str] = None
    type: str
    content: Dict[str, Any]
    source: Dict[str, str]
    confidence: float = Field(ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class KnowledgeQueryParams(BaseModel):
    """Model for knowledge query parameters."""
    type: Optional[str] = None
    confidence_min: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    text_search: Optional[str] = None
    source: Optional[str] = None
    limit: Optional[int] = Field(default=100, ge=1, le=1000)
    offset: Optional[int] = Field(default=0, ge=0)


class KnowledgeUpdateRequest(BaseModel):
    """Model for knowledge item update request."""
    content: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    metadata: Optional[Dict[str, Any]] = None


class BookMetadata(BaseModel):
    """Model for book metadata."""
    title: Optional[str] = None
    author: Optional[str] = None
    publication_year: Optional[int] = None
    publisher: Optional[str] = None
    isbn: Optional[str] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None


# Create router
router = APIRouter(
    prefix="/api/knowledge",
    tags=["knowledge"],
    dependencies=[Depends(api_key_auth)]
)

# Initialize service
knowledge_service = KnowledgeEngineService()
logger = logging.getLogger(__name__)


@router.get("/", response_model=List[KnowledgeItem])
async def query_knowledge(query: KnowledgeQueryParams = Depends()):
    """Query the knowledge base.
    
    Args:
        query: Query parameters
        
    Returns:
        List of matching knowledge items
    """
    try:
        # Convert Pydantic model to dict
        query_dict = query.dict(exclude_none=True)
        
        # Add limit and offset for pagination
        limit = query_dict.pop('limit', 100)
        offset = query_dict.pop('offset', 0)
        
        # Query the knowledge base
        results = knowledge_service.query_knowledge(query_dict)
        
        # Apply pagination
        paginated_results = results[offset:offset + limit]
        
        return paginated_results
    except Exception as e:
        logger.error(f"Error querying knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics", response_model=Dict[str, Any])
async def get_knowledge_statistics():
    """Get statistics about the knowledge base.
    
    Returns:
        Dictionary with knowledge base statistics
    """
    try:
        return knowledge_service.get_knowledge_statistics()
    except Exception as e:
        logger.error(f"Error getting knowledge statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{item_id}", response_model=KnowledgeItem)
async def get_knowledge_item(item_id: str):
    """Get a specific knowledge item by ID.
    
    Args:
        item_id: ID of the knowledge item
        
    Returns:
        Knowledge item
    """
    try:
        # Query for the specific item
        results = knowledge_service.query_knowledge({'id': item_id})
        
        if not results:
            raise HTTPException(status_code=404, detail=f"Knowledge item with ID {item_id} not found")
        
        return results[0]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting knowledge item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/", response_model=Dict[str, Any])
async def create_knowledge_items(items: List[KnowledgeItem]):
    """Create new knowledge items.
    
    Args:
        items: List of knowledge items to create
        
    Returns:
        Dictionary with creation results
    """
    try:
        # Convert Pydantic models to dicts
        item_dicts = [item.dict(exclude_none=True) for item in items]
        
        # Save the items
        result = knowledge_service.save_knowledge_items(item_dicts)
        
        if not result.get('success', False):
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating knowledge items: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/{item_id}", response_model=Dict[str, Any])
async def update_knowledge_item(item_id: str, update_request: KnowledgeUpdateRequest):
    """Update a knowledge item.
    
    Args:
        item_id: ID of the knowledge item to update
        update_request: Update request
        
    Returns:
        Dictionary with update results
    """
    try:
        # Convert Pydantic model to dict
        updates = update_request.dict(exclude_none=True)
        
        # Update the item
        result = knowledge_service.update_knowledge_item(item_id, updates)
        
        if not result.get('success', False):
            if 'not found' in result.get('error', '').lower():
                raise HTTPException(status_code=404, detail=f"Knowledge item with ID {item_id} not found")
            else:
                raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating knowledge item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{item_id}", response_model=Dict[str, Any])
async def delete_knowledge_item(item_id: str):
    """Delete a knowledge item.
    
    Args:
        item_id: ID of the knowledge item to delete
        
    Returns:
        Dictionary with deletion results
    """
    try:
        # Delete the item
        result = knowledge_service.delete_knowledge_item(item_id)
        
        if not result.get('success', False):
            if 'not found' in result.get('error', '').lower():
                raise HTTPException(status_code=404, detail=f"Knowledge item with ID {item_id} not found")
            else:
                raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting knowledge item: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/book", response_model=Dict[str, Any])
async def extract_knowledge_from_book(
    book_file: UploadFile = File(...),
    metadata: Optional[BookMetadata] = Body(None)
):
    """Extract knowledge from a book file.
    
    Args:
        book_file: Book file to extract knowledge from
        metadata: Optional book metadata
        
    Returns:
        Dictionary with extraction results
    """
    try:
        # Create a temporary file to store the uploaded content
        temp_file_path = f"temp_{book_file.filename}"
        
        try:
            # Save the uploaded file
            with open(temp_file_path, "wb") as temp_file:
                content = await book_file.read()
                temp_file.write(content)
            
            # Convert Pydantic model to dict if provided
            metadata_dict = metadata.dict(exclude_none=True) if metadata else None
            
            # Extract knowledge from the book
            result = knowledge_service.extract_knowledge_from_book(temp_file_path, metadata_dict)
            
            if not result.get('success', False):
                raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
            
            return result
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error extracting knowledge from book: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract/batch", response_model=Dict[str, Any])
async def batch_process_books(directory_path: str = Query(..., description="Path to directory containing book files")):
    """Process all book files in a directory.
    
    Args:
        directory_path: Path to directory containing book files
        
    Returns:
        Dictionary with processing results
    """
    try:
        # Process the books in the directory
        result = knowledge_service.batch_process_books(directory_path)
        
        if not result.get('success', False) and 'error' in result:
            raise HTTPException(status_code=500, detail=result.get('error', 'Unknown error'))
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error batch processing books: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))