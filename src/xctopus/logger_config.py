"""
Centralized logging configuration for Xctopus.

Logging system with:
- FileHandler: DEBUG level (all information to file)
- ConsoleHandler: WARNING level (only warnings/errors in console)
- Automatic file rotation
- Error deduplication for console (avoids spam)
- stderr interception to capture external library errors
"""

import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from collections import OrderedDict

# Default values (can be overridden from settings)
_DEFAULT_LOG_DIR = Path("logs")
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "xctopus.log"
_DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_LOG_BACKUP_COUNT = 5
_DEFAULT_LOG_LEVEL_FILE = "DEBUG"
_DEFAULT_LOG_LEVEL_CONSOLE = "WARNING"

# Deduplication cache (shared state for filter function)
_dedup_cache: OrderedDict[str, float] = OrderedDict()
_dedup_last_cleanup = time.time()
_DEDUP_CACHE_SIZE = 10
_DEDUP_WINDOW_SECONDS = 60


def _deduplication_filter(record: logging.LogRecord) -> bool:
    """
    Filter function to prevent duplicate error messages in console.
    
    Returns:
        True if message should be shown, False if duplicate
    """
    global _dedup_cache, _dedup_last_cleanup
    
    # Only filter ERROR and WARNING levels (console shows these)
    if record.levelno < logging.WARNING:
        return True
    
    # Create message key (level + message text)
    message_key = f"{record.levelname}:{record.getMessage()}"
    current_time = time.time()
    
    # Cleanup old entries periodically
    if current_time - _dedup_last_cleanup > _DEDUP_WINDOW_SECONDS:
        cutoff_time = current_time - _DEDUP_WINDOW_SECONDS
        keys_to_remove = [
            key for key, timestamp in _dedup_cache.items()
            if timestamp < cutoff_time
        ]
        for key in keys_to_remove:
            _dedup_cache.pop(key, None)
        _dedup_last_cleanup = current_time
    
    # Check if message is duplicate
    if message_key in _dedup_cache:
        # Duplicate found within window - suppress console output
        return False
    
    # New message - add to cache and allow console output
    _dedup_cache[message_key] = current_time
    
    # Limit cache size (remove oldest if full)
    if len(_dedup_cache) > _DEDUP_CACHE_SIZE:
        _dedup_cache.popitem(last=False)
    
    return True


# stderr interception state
_stderr_buffer = ""
_original_stderr = None


def _stderr_write_interceptor(text: str, logger: logging.Logger):
    """Intercept stderr writes and redirect to logger."""
    global _stderr_buffer
    
    if text:
        _stderr_buffer += text
        # Log complete lines
        if '\n' in _stderr_buffer:
            lines = _stderr_buffer.split('\n')
            _stderr_buffer = lines[-1]  # Keep incomplete line in buffer
            for line in lines[:-1]:
                if line.strip():
                    # Filter out tqdm progress bars (they write to stderr but aren't errors)
                    # tqdm lines typically contain: "|", "%", "embedding/s", "KNs=", "Buffers="
                    is_tqdm_output = (
                        "Processing embeddings" in line or
                        "Procesando embeddings" in line or  # Keep old Spanish for backward compatibility
                        "|" in line and ("%" in line or "embedding" in line.lower() or "KNs=" in line or "Buffers=" in line) or
                        (line.strip().startswith("Processing") or line.strip().startswith("Procesando")) and ("%" in line or "/" in line)
                    )
                    
                    if not is_tqdm_output:
                        # Log to file (only non-tqdm output)
                        logger.error(f"[stderr] {line.strip()}")
                    # Always write to original stderr for compatibility (tqdm needs this)
                    if _original_stderr:
                        _original_stderr.write(f"{line}\n")


def _stderr_flush_interceptor(logger: logging.Logger):
    """Flush stderr buffer."""
    global _stderr_buffer
    
    if _stderr_buffer.strip():
        # Filter out tqdm progress bars (they write to stderr but aren't errors)
        is_tqdm_output = (
            "Processing embeddings" in _stderr_buffer or
            "Procesando embeddings" in _stderr_buffer or  # Keep old Spanish for backward compatibility
            "|" in _stderr_buffer and ("%" in _stderr_buffer or "embedding" in _stderr_buffer.lower() or "KNs=" in _stderr_buffer or "Buffers=" in _stderr_buffer) or
            (_stderr_buffer.strip().startswith("Processing") or _stderr_buffer.strip().startswith("Procesando")) and ("%" in _stderr_buffer or "/" in _stderr_buffer)
        )
        
        if not is_tqdm_output:
            logger.error(f"[stderr] {_stderr_buffer.strip()}")
        if _original_stderr:
            _original_stderr.write(_stderr_buffer)
            _stderr_buffer = ""
            _original_stderr.flush()


def _custom_excepthook(exc_type, exc_value, exc_traceback):
    """
    Custom exception handler to log unhandled exceptions.
    
    Logs complete traceback to file, shows summary in console.
    """
    logger = logging.getLogger("xctopus")
    
    # Log complete exception to file
    logger.error(
        f"Unhandled exception: {exc_type.__name__}: {exc_value}",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    
    # Show only summary in console (will go through deduplication filter)
    logger.error(f"{exc_type.__name__}: {exc_value}")
    
    # Call original excepthook if it exists (for default Python behavior)
    if hasattr(sys, '__excepthook__') and sys.__excepthook__ != _custom_excepthook:
        sys.__excepthook__(exc_type, exc_value, exc_traceback)


def setup_logging(
    log_dir: Path = None,
    log_file: Path = None,
    log_max_bytes: int = None,
    log_backup_count: int = None,
    log_level_file: str = None,
    log_level_console: str = None,
) -> None:
    """
    Configure centralized logging system.
    
    Creates:
    - FileHandler with DEBUG level (everything to file)
    - ConsoleHandler with WARNING level (only warnings/errors in console)
    - Structured format with timestamp, level, module, message
    - Automatic file rotation
    
    Args:
        log_dir: Directory for logs (default: from settings or "logs")
        log_file: Log file (default: from settings or "logs/xctopus.log")
        log_max_bytes: Maximum size per file (default: 10 MB)
        log_backup_count: Number of backups (default: 5)
        log_level_file: Level for file (default: "DEBUG")
        log_level_console: Level for console (default: "WARNING")
    """
    # Try to import from settings, use defaults if it fails
    try:
        from .settings import (
            LOG_DIR as SETTINGS_LOG_DIR,
            LOG_FILE as SETTINGS_LOG_FILE,
            LOG_MAX_BYTES as SETTINGS_LOG_MAX_BYTES,
            LOG_BACKUP_COUNT as SETTINGS_LOG_BACKUP_COUNT,
            LOG_LEVEL_FILE as SETTINGS_LOG_LEVEL_FILE,
            LOG_LEVEL_CONSOLE as SETTINGS_LOG_LEVEL_CONSOLE,
        )
        _log_dir = log_dir or SETTINGS_LOG_DIR
        _log_file = log_file or SETTINGS_LOG_FILE
        _log_max_bytes = log_max_bytes or SETTINGS_LOG_MAX_BYTES
        _log_backup_count = log_backup_count or SETTINGS_LOG_BACKUP_COUNT
        _log_level_file = log_level_file or SETTINGS_LOG_LEVEL_FILE
        _log_level_console = log_level_console or SETTINGS_LOG_LEVEL_CONSOLE
    except ImportError:
        # Usar defaults si settings no est? disponible
        _log_dir = log_dir or _DEFAULT_LOG_DIR
        _log_file = log_file or _DEFAULT_LOG_FILE
        _log_max_bytes = log_max_bytes or _DEFAULT_LOG_MAX_BYTES
        _log_backup_count = log_backup_count or _DEFAULT_LOG_BACKUP_COUNT
        _log_level_file = log_level_file or _DEFAULT_LOG_LEVEL_FILE
        _log_level_console = log_level_console or _DEFAULT_LOG_LEVEL_CONSOLE
    
    # Crear directorio de logs si no existe
    _log_dir.mkdir(exist_ok=True)
    
    # Get main logger
    logger = logging.getLogger("xctopus")
    logger.setLevel(logging.DEBUG)
    
    # Avoid duplicating handlers if they're already configured
    if logger.handlers:
        return
    
    # ========================================================================
    # FileHandler: Everything to file (DEBUG)
    # ========================================================================
    
    file_handler = RotatingFileHandler(
        _log_file,
        maxBytes=_log_max_bytes,
        backupCount=_log_backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, _log_level_file))
    
    # Format for file: timestamp | level | module | message
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # ========================================================================
    # ConsoleHandler: Only warnings/errors (WARNING)
    # ========================================================================
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, _log_level_console))
    
    # Format for console: timestamp | level | message (more compact)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add deduplication filter to console handler (prevents spam)
    console_handler.addFilter(_deduplication_filter)
    
    logger.addHandler(console_handler)
    
    # ========================================================================
    # Intercept stderr to capture external library errors
    # ========================================================================
    # Capture errors from PyTorch, transformers, etc. that write to stderr
    global _original_stderr
    _original_stderr = sys.stderr
    
    # Create wrapper object for stderr that intercepts writes
    class StderrWrapper:
        def write(self, text: str):
            _stderr_write_interceptor(text, logger)
            if _original_stderr:
                _original_stderr.write(text)
        
        def flush(self):
            _stderr_flush_interceptor(logger)
            if _original_stderr:
                _original_stderr.flush()
        
        def __getattr__(self, name):
            return getattr(_original_stderr, name)
    
    sys.stderr = StderrWrapper()
    
    # ========================================================================
    # Intercept unhandled exceptions
    # ========================================================================
    # Capture exceptions that aren't caught by try/except blocks
    sys.excepthook = _custom_excepthook
    
    # Initial log
    logger.info("Logging system configured")
    logger.debug(f"FileHandler: {_log_file} (level {_log_level_file})")
    logger.debug(f"ConsoleHandler: stdout (level {_log_level_console})")
    logger.debug("Message deduplication filter and stderr interception enabled")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
    
    Returns:
        Configured logger
    """
    return logging.getLogger(name)
