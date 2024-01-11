#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging configuration file. Log messages are written to a file and printed to the console.

Usage:
    from log_setup import logger
    logger.info("This is a log message.")
    
"""

import logging

# Create a logger
logger = logging.getLogger("logger")

# Set the level of the logger
logger.setLevel(logging.DEBUG)

# Create a file handler
file_handler = logging.FileHandler("../logs/logfile.log")

# Create a stream handler
stream_handler = logging.StreamHandler()

# Create a formatter and add it to the handlers
format_log = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
format_console = logging.Formatter("%(levelname)s - %(message)s")

file_handler.setFormatter(format_log)
stream_handler.setFormatter(format_console)

# Remove all handlers from the logger so that the log messages are not duplicated
logger.handlers = []

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
