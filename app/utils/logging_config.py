"""
Production Logging Configuration
Structured logging for production deployment
"""
import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """
    JSON formatter for structured logging in production.
    Compatible with log aggregation tools (ELK, CloudWatch, etc.)
    """
    
    def format(self, record: logging.LogRecord) -> str:
        log_data: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        if hasattr(record, "call_sid"):
            log_data["call_sid"] = record.call_sid
        if hasattr(record, "caller_number"):
            log_data["caller_number"] = record.caller_number
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        
        return json.dumps(log_data)


def setup_production_logging(
    level: str = "INFO",
    json_format: bool = True
) -> None:
    """
    Configure logging for production environment.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: Use JSON format for structured logging
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
    
    root_logger.addHandler(console_handler)
    
    # Reduce noise from third-party libraries
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("aiosqlite").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


class CallLogger:
    """
    Context-aware logger for tracking individual calls.
    """
    
    def __init__(self, call_sid: str, caller_number: str = None):
        self.call_sid = call_sid
        self.caller_number = caller_number
        self.logger = logging.getLogger("call_tracker")
    
    def _make_record(self, level: int, msg: str) -> None:
        record = self.logger.makeRecord(
            self.logger.name,
            level,
            "", 0, msg, (), None
        )
        record.call_sid = self.call_sid
        record.caller_number = self.caller_number
        self.logger.handle(record)
    
    def info(self, msg: str) -> None:
        self._make_record(logging.INFO, msg)
    
    def warning(self, msg: str) -> None:
        self._make_record(logging.WARNING, msg)
    
    def error(self, msg: str) -> None:
        self._make_record(logging.ERROR, msg)


# Initialize production logging when imported
def init_logging():
    """Initialize logging based on environment"""
    import os
    
    env = os.getenv("ENVIRONMENT", "development")
    debug = os.getenv("DEBUG", "true").lower() == "true"
    
    if env == "production":
        setup_production_logging(level="INFO", json_format=True)
    else:
        setup_production_logging(level="DEBUG" if debug else "INFO", json_format=False)
