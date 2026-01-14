"""Logging configuration for encoder-lean."""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "encoder_lean",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    console_output: bool = True
) -> logging.Logger:
    """
    Set up a logger with file and/or console output.
    
    Args:
        name: Logger name
        log_file: Path to log file (if None, no file logging)
        log_level: Logging level (default: INFO)
        console_output: Whether to output to console (default: True)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_default_log_file() -> str:
    """
    Get default log file path with timestamp.
    
    Returns:
        Path to log file
    """
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(logs_dir / f"encoder_lean_{timestamp}.log")
