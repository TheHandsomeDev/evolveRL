import logging
from typing import Optional

VERBOSE_LEVELS = {
    0: logging.WARNING,  # Default
    1: logging.INFO,     # -v
    2: logging.DEBUG,    # -vv
    3: 5,               # -vvv (Custom detailed level)
    4: 4,               # -vvvv (Custom very detailed level)
    5: 3                # -vvvvv (Custom most detailed level)
}

def setup_logging(verbose_level: int = 0) -> None:
    """Configure logging based on verbosity level"""
    # Add custom logging levels
    logging.addLevelName(5, "DETAILED")
    logging.addLevelName(4, "VERY_DETAILED")
    logging.addLevelName(3, "MOST_DETAILED")
    
    # Set logging level based on verbosity
    level = VERBOSE_LEVELS.get(verbose_level, logging.WARNING)
    
    # Configure logging format based on verbosity
    if verbose_level >= 3:
        format_str = '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
    elif verbose_level >= 1:
        format_str = '%(levelname)s | %(message)s'
    else:
        format_str = '%(message)s'
    
    logging.basicConfig(
        level=level,
        format=format_str,
        datefmt='%Y-%m-%d %H:%M:%S'
    ) 