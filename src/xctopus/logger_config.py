"""
Centralized logging configuration for Xctopus.

Logging system with:
- FileHandler: DEBUG level (all information to file)
- ConsoleHandler: WARNING level (only warnings/errors in console)
- Automatic file rotation
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Default values (can be overridden from settings)
_DEFAULT_LOG_DIR = Path("logs")
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "xctopus.log"
_DEFAULT_LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_DEFAULT_LOG_BACKUP_COUNT = 5
_DEFAULT_LOG_LEVEL_FILE = "DEBUG"
_DEFAULT_LOG_LEVEL_CONSOLE = "WARNING"


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
        log_file: Archivo de log (default: desde settings o "logs/xctopus.log")
        log_max_bytes: Tama?o m?ximo por archivo (default: 10 MB)
        log_backup_count: N?mero de backups (default: 5)
        log_level_file: Nivel para archivo (default: "DEBUG")
        log_level_console: Nivel para consola (default: "WARNING")
    """
    # Intentar importar desde settings, usar defaults si falla
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
    
    # Obtener logger principal
    logger = logging.getLogger("xctopus")
    logger.setLevel(logging.DEBUG)
    
    # Evitar duplicar handlers si ya est?n configurados
    if logger.handlers:
        return
    
    # ========================================================================
    # FileHandler: Todo al archivo (DEBUG)
    # ========================================================================
    
    file_handler = RotatingFileHandler(
        _log_file,
        maxBytes=_log_max_bytes,
        backupCount=_log_backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(getattr(logging, _log_level_file))
    
    # Formato para archivo: timestamp | level | module | message
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # ========================================================================
    # ConsoleHandler: Solo warnings/errores (WARNING)
    # ========================================================================
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, _log_level_console))
    
    # Formato para consola: timestamp | level | message (m?s compacto)
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # Log inicial
    logger.info("Sistema de logging configurado")
    logger.debug(f"FileHandler: {_log_file} (nivel {_log_level_file})")
    logger.debug(f"ConsoleHandler: stdout (nivel {_log_level_console})")


def get_logger(name: str) -> logging.Logger:
    """
    Obtiene un logger con el nombre especificado.
    
    Args:
        name: Nombre del logger (t?picamente __name__)
    
    Returns:
        Logger configurado
    """
    return logging.getLogger(name)
