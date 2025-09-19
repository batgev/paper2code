"""
Main Paper2Code Processor

This module provides the primary interface for processing research papers
and generating code implementations.
"""

import asyncio
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union
from dataclasses import dataclass

from .config.manager import ConfigManager
from .workflows.orchestrator import WorkflowOrchestrator
from .utils.file_processor import FileProcessor
from .utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of paper processing operation"""
    success: bool
    output_path: Optional[str] = None
    files: Optional[list] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class Paper2CodeProcessor:
    """
    Main processor for converting research papers to code implementations.
    
    This class orchestrates the entire pipeline from paper analysis to code generation.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the processor.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.orchestrator = WorkflowOrchestrator(self.config)
        self.file_processor = FileProcessor()
        
        # Ensure output directory exists
        self._setup_output_directory()
    
    def _setup_output_directory(self):
        """Setup the output directory for generated code"""
        try:
            output_dir = Path(self.config.get('output.base_directory', './output'))
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory setup: {output_dir}")
        except PermissionError:
            # Fallback to user temp directory
            import tempfile
            output_dir = Path(tempfile.gettempdir()) / "paper2code_output"
            output_dir.mkdir(parents=True, exist_ok=True)
            self.config.set('output.base_directory', str(output_dir))
            logger.warning(f"Using temp output directory due to permissions: {output_dir}")
    
    async def process_paper(
        self,
        input_source: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        mode: str = "comprehensive",
        **kwargs
    ) -> ProcessingResult:
        """
        Process a research paper and generate code implementation.
        
        Args:
            input_source: Path to paper file or URL
            output_dir: Optional custom output directory
            mode: Processing mode ("comprehensive" or "fast")
            **kwargs: Additional processing options
            
        Returns:
            ProcessingResult with success status and details
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚀 Starting paper processing: {input_source}")
            logger.info(f"📊 Processing mode: {mode}")
            
            # Validate input
            input_path = await self._validate_input(input_source)
            
            # Setup output directory
            output_path = self._setup_paper_output_directory(input_path, output_dir)
            
            # Configure processing options
            processing_options = {
                "mode": mode,
                "enable_repository_search": mode == "comprehensive",
                "enable_segmentation": self.config.get('document_segmentation.enabled', True),
                **kwargs
            }

            # Optionally override LLM provider/model
            llm_provider = kwargs.get('llm_provider')
            llm_model = kwargs.get('llm_model')
            if llm_provider:
                self.config.set('llm.preferred_provider', llm_provider)
            if llm_model:
                if (self.config.get('llm.preferred_provider') or '').lower() == 'ollama':
                    self.config.set('llm.ollama.default_model', llm_model)
            
            # Process the paper
            # Optional progress callback propagation
            progress_callback = kwargs.get('progress_callback')
            result = await self.orchestrator.process_paper(
                input_path=input_path,
                output_path=output_path,
                options=processing_options,
                progress_callback=progress_callback
            )
            
            processing_time = time.time() - start_time
            
            if result["status"] == "success":
                logger.info(f"✅ Processing completed successfully in {processing_time:.2f}s")
                logger.info(f"📁 Output directory: {result['output_path']}")
                
                return ProcessingResult(
                    success=True,
                    output_path=result["output_path"],
                    files=result.get("generated_files", []),
                    processing_time=processing_time,
                    metadata=result.get("metadata", {})
                )
            else:
                error_msg = result.get("error", "Unknown processing error")
                logger.error(f"❌ Processing failed: {error_msg}")
                
                return ProcessingResult(
                    success=False,
                    error=error_msg,
                    processing_time=processing_time
                )
                
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Processing error: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            return ProcessingResult(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    async def _validate_input(self, input_source: Union[str, Path]) -> Path:
        """
        Validate and process the input source.
        
        Args:
            input_source: File path or URL
            
        Returns:
            Validated Path object
            
        Raises:
            ValueError: If input is invalid
        """
        if isinstance(input_source, str):
            # Handle URLs
            if input_source.startswith(('http://', 'https://')):
                logger.info(f"📥 Downloading paper from URL: {input_source}")
                # Download and convert URL to local file
                input_path = await self.file_processor.download_from_url(input_source)
                logger.info(f"📁 Downloaded to: {input_path}")
                return input_path
            else:
                input_source = Path(input_source)
        
        if not input_source.exists():
            raise ValueError(f"Input file does not exist: {input_source}")
        
        if not input_source.is_file():
            raise ValueError(f"Input must be a file, not directory: {input_source}")
        
        # Check supported file types
        supported_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.html'}
        if input_source.suffix.lower() not in supported_extensions:
            logger.warning(f"⚠️ Unsupported file type: {input_source.suffix}")
            logger.info(f"📋 Supported types: {', '.join(supported_extensions)}")
        
        return input_source
    
    def _setup_paper_output_directory(
        self,
        input_path: Path,
        custom_output_dir: Optional[Union[str, Path]] = None
    ) -> Path:
        """
        Setup output directory for a specific paper using extracted title.
        
        Args:
            input_path: Path to the input paper
            custom_output_dir: Optional custom output directory
            
        Returns:
            Path to the paper's output directory
        """
        if custom_output_dir:
            output_base = Path(custom_output_dir)
        else:
            output_base = Path(self.config.get('output.base_directory', './output'))
        
        # Try to extract paper title for better naming
        paper_name = input_path.stem
        
        if input_path.suffix.lower() == '.pdf':
            try:
                metadata = self.file_processor.extract_pdf_title_and_metadata(input_path)
                if metadata['title']:
                    paper_name = metadata['title']
                    logger.info(f"📄 Using extracted PDF title: {paper_name}")
                else:
                    logger.info(f"📄 No title found in PDF metadata, using filename")
            except Exception as e:
                logger.warning(f"⚠️ Title extraction failed: {e}, using filename")
        
        # Clean paper name for directory use
        paper_name = "".join(c for c in paper_name if c.isalnum() or c in (' ', '-', '_', '.')).strip()
        paper_name = paper_name.replace(' ', '_').replace('.', '_')
        
        # Limit length and remove common suffixes
        paper_name = paper_name[:100]  # Max 100 chars
        paper_name = re.sub(r'_+(pdf|doc|docx|txt|md)$', '', paper_name, flags=re.IGNORECASE)
        
        # Add timestamp if directory exists to avoid conflicts
        paper_output_dir = output_base / paper_name
        if paper_output_dir.exists():
            timestamp = str(int(time.time()))
            paper_output_dir = output_base / f"{paper_name}_{timestamp}"
        
        paper_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Paper output directory: {paper_output_dir}")
        
        return paper_output_dir
    
    async def list_recent_outputs(self, limit: int = 10) -> list:
        """
        List recent processing outputs.
        
        Args:
            limit: Maximum number of outputs to return
            
        Returns:
            List of recent output directories with metadata
        """
        output_base = Path(self.config.get('output.base_directory', './output'))
        
        if not output_base.exists():
            return []
        
        # Get all output directories
        output_dirs = [
            d for d in output_base.iterdir() 
            if d.is_dir() and not d.name.startswith('.')
        ]
        
        # Sort by modification time (newest first)
        output_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Return limited results with metadata
        results = []
        for output_dir in output_dirs[:limit]:
            try:
                stat = output_dir.stat()
                results.append({
                    "name": output_dir.name,
                    "path": str(output_dir),
                    "created": stat.st_ctime,
                    "modified": stat.st_mtime,
                    "size_mb": sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file()) / 1024 / 1024
                })
            except Exception as e:
                logger.warning(f"Error reading output directory {output_dir}: {e}")
        
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        return self.config.to_dict()
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration"""
        self.config.update(updates)
        logger.info("Configuration updated")


# Convenience functions for direct usage
async def process_paper_file(
    file_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    mode: str = "comprehensive"
) -> ProcessingResult:
    """
    Convenience function to process a single paper file.
    
    Args:
        file_path: Path to the paper file
        output_dir: Optional output directory
        mode: Processing mode
        
    Returns:
        ProcessingResult
    """
    processor = Paper2CodeProcessor()
    return await processor.process_paper(file_path, output_dir, mode)


async def process_paper_url(
    url: str,
    output_dir: Optional[Union[str, Path]] = None,
    mode: str = "comprehensive"
) -> ProcessingResult:
    """
    Convenience function to process a paper from URL.
    
    Args:
        url: URL to the paper
        output_dir: Optional output directory  
        mode: Processing mode
        
    Returns:
        ProcessingResult
    """
    processor = Paper2CodeProcessor()
    return await processor.process_paper(url, output_dir, mode)


# Synchronous wrappers for non-async usage
def process_paper_sync(
    input_source: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    mode: str = "comprehensive"
) -> ProcessingResult:
    """
    Synchronous wrapper for paper processing.
    
    Args:
        input_source: Path to paper or URL
        output_dir: Optional output directory
        mode: Processing mode
        
    Returns:
        ProcessingResult
    """
    processor = Paper2CodeProcessor()
    return asyncio.run(processor.process_paper(input_source, output_dir, mode))
