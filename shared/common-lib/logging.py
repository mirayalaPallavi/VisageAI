import logging
import sys
import json
from datetime import datetime
from typing import Any, Dict, Optional
import structlog
from .config import get_config

def setup_logging(
    level: str = "INFO",
    format_type: str = "json",
    output_file: Optional[str] = None,
    service_name: str = "unknown"
) -> None:
    """
    Setup structured logging for the application
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Log format type (json, text)
        output_file: Optional output file path
        service_name: Name of the service for logging context
    """
    # Configure structlog
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    if format_type == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())
    
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard library logging
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    if format_type == "json":
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if output_file:
        file_handler = logging.FileHandler(output_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Add service context to all loggers
    structlog.configure(
        processors=processors + [
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

def get_logger(name: str = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)

def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log function call with parameters
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger = get_logger()
    logger.info("Function call", function=func_name, parameters=kwargs)

def log_function_result(func_name: str, result: Any, execution_time: float = None) -> None:
    """
    Log function execution result
    
    Args:
        func_name: Name of the function
        result: Function result
        execution_time: Execution time in seconds
    """
    logger = get_logger()
    log_data = {
        "function": func_name,
        "result_type": type(result).__name__,
        "execution_time": execution_time
    }
    
    # Add result size if applicable
    if hasattr(result, '__len__'):
        log_data["result_size"] = len(result)
    
    logger.info("Function result", **log_data)

def log_error(error: Exception, context: Dict[str, Any] = None) -> None:
    """
    Log error with context
    
    Args:
        error: Exception that occurred
        context: Additional context information
    """
    logger = get_logger()
    
    error_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "error_traceback": getattr(error, '__traceback__', None)
    }
    
    if context:
        error_data.update(context)
    
    logger.error("Error occurred", **error_data)

def log_performance(operation: str, duration: float, **kwargs) -> None:
    """
    Log performance metrics
    
    Args:
        operation: Name of the operation
        duration: Duration in seconds
        **kwargs: Additional performance metrics
    """
    logger = get_logger()
    
    performance_data = {
        "operation": operation,
        "duration_seconds": duration,
        "duration_ms": duration * 1000
    }
    performance_data.update(kwargs)
    
    logger.info("Performance metric", **performance_data)

def log_security_event(event_type: str, user_id: str = None, ip_address: str = None, **kwargs) -> None:
    """
    Log security-related events
    
    Args:
        event_type: Type of security event
        user_id: ID of the user involved
        ip_address: IP address of the request
        **kwargs: Additional security event data
    """
    logger = get_logger()
    
    security_data = {
        "event_type": event_type,
        "user_id": user_id,
        "ip_address": ip_address,
        "timestamp": datetime.utcnow().isoformat()
    }
    security_data.update(kwargs)
    
    logger.warning("Security event", **security_data)

def log_database_operation(operation: str, table: str, duration: float = None, **kwargs) -> None:
    """
    Log database operations
    
    Args:
        operation: Database operation type (SELECT, INSERT, UPDATE, DELETE)
        table: Table name
        duration: Operation duration in seconds
        **kwargs: Additional database operation data
    """
    logger = get_logger()
    
    db_data = {
        "operation": operation,
        "table": table,
        "duration_seconds": duration
    }
    db_data.update(kwargs)
    
    logger.info("Database operation", **db_data)

def log_api_request(
    method: str,
    path: str,
    status_code: int,
    duration: float,
    user_id: str = None,
    ip_address: str = None,
    **kwargs
) -> None:
    """
    Log API request details
    
    Args:
        method: HTTP method
        path: Request path
        status_code: HTTP status code
        duration: Request duration in seconds
        user_id: ID of the authenticated user
        ip_address: IP address of the request
        **kwargs: Additional request data
    """
    logger = get_logger()
    
    request_data = {
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_seconds": duration,
        "duration_ms": duration * 1000,
        "user_id": user_id,
        "ip_address": ip_address
    }
    request_data.update(kwargs)
    
    # Log level based on status code
    if status_code >= 500:
        logger.error("API request error", **request_data)
    elif status_code >= 400:
        logger.warning("API request warning", **request_data)
    else:
        logger.info("API request", **request_data)

# Initialize logging with config
def init_logging(service_name: str = "unknown") -> None:
    """
    Initialize logging with configuration from config file
    
    Args:
        service_name: Name of the service
    """
    config = get_config()
    setup_logging(
        level=config.logging.level,
        format_type=config.logging.format,
        output_file=config.logging.output_file,
        service_name=service_name
    )
