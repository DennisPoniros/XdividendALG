"""
Centralized logging configuration for Dividend Capture Algorithm
Provides structured logging with file and console output
"""

import logging
import os
from datetime import datetime
from typing import Optional


class AlgoLogger:
    """
    Centralized logger for the dividend capture algorithm
    """

    _loggers = {}

    @staticmethod
    def get_logger(name: str, log_dir: str = '/mnt/user-data/logs') -> logging.Logger:
        """
        Get or create a logger with the specified name

        Args:
            name: Name of the logger (usually module name)
            log_dir: Directory for log files

        Returns:
            Configured logger instance
        """
        if name in AlgoLogger._loggers:
            return AlgoLogger._loggers[name]

        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)

        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        # Avoid duplicate handlers
        if logger.handlers:
            return logger

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        simple_formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        )

        # File handler (detailed logs)
        log_filename = f"{log_dir}/dividend_algo_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)

        # Console handler (simplified output)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)

        # Add handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Cache logger
        AlgoLogger._loggers[name] = logger

        return logger

    @staticmethod
    def set_level(level: str):
        """
        Set logging level for all loggers

        Args:
            level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        """
        level_map = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }

        log_level = level_map.get(level.upper(), logging.INFO)

        for logger in AlgoLogger._loggers.values():
            logger.setLevel(log_level)
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setLevel(log_level)


# Convenience function for quick logger creation
def get_logger(name: str) -> logging.Logger:
    """Get logger instance for module"""
    return AlgoLogger.get_logger(name)


if __name__ == '__main__':
    # Test logging
    logger = get_logger('test')

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    print("\n✅ Logging framework test complete")
