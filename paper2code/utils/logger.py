"""
Logging utilities for Paper2Code
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path
from typing import Optional


class SafeFormatter(logging.Formatter):
    """
    Formatter that strips characters not encodable by the target stream encoding
    to prevent UnicodeEncodeError on Windows consoles.
    """

    def __init__(self, fmt: str, target_encoding: Optional[str] = None):
        super().__init__(fmt)
        self._encoding = target_encoding or 'utf-8'

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        try:
            message.encode(self._encoding, errors='strict')
            return message
        except Exception:
            return message.encode(self._encoding, errors='ignore').decode(self._encoding, errors='ignore')


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration for Paper2Code.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
        console: Whether to log to console
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("paper2code")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Default format
    if not format_string:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console:
        # Ensure stdout uses UTF-8 when possible; if not, strip unencodable chars
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        try:
            sys.stdout.reconfigure(encoding='utf-8')  # type: ignore[attr-defined]
            console_handler.setFormatter(logging.Formatter(format_string))
        except Exception:
            enc = getattr(sys.stdout, 'encoding', 'utf-8') or 'utf-8'
            console_handler.setFormatter(SafeFormatter(format_string, target_encoding=enc))
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler to prevent large log files
            file_handler = logging.handlers.RotatingFileHandler(
                log_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'  # Force UTF-8 encoding
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(logging.Formatter(format_string))
            logger.addHandler(file_handler)
        except Exception:
            # Skip file logging if it fails
            pass
    
    return logger


def get_logger(name: str = None) -> logging.Logger:
    """
    Get a logger instance.
    
    Args:
        name: Logger name (defaults to paper2code)
        
    Returns:
        Logger instance
    """
    if name:
        return logging.getLogger(f"paper2code.{name}")
    return logging.getLogger("paper2code")


# Initialize default logging
_default_logger = None

def init_default_logging():
    """Initialize default logging if not already done"""
    global _default_logger
    
    if _default_logger is None:
        # Get log level from environment
        log_level = os.getenv("PAPER2CODE_LOG_LEVEL", "INFO")
        
        # Setup log file path - use temp dir if current dir not writable
        try:
            log_dir = Path(os.getenv("PAPER2CODE_LOG_DIR", "./logs"))
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "paper2code.log"
        except (PermissionError, OSError):
            # Fallback to temp directory if current dir not writable
            import tempfile
            log_dir = Path(tempfile.gettempdir()) / "paper2code"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "paper2code.log"
        
        _default_logger = setup_logging(
            level=log_level,
            log_file=str(log_file),
            console=True
        )
    
    return _default_logger


# Initialize on import
init_default_logging()
