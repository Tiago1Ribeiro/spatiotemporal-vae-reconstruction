#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logging configuration file.
"""

import logging

# Create a logger
logger = logging.getLogger("logger")

# Set the level of the logger
logger.setLevel(logging.INFO)

# Create a file handler
file_handler = logging.FileHandler("../logs/logfile.log")

# Create a stream handler
stream_handler = logging.StreamHandler()

# Create a formatter and add it to the handlers
formatter = logging.Formatter(
    "%(asctime)s - %(name)s -12s %(levelname)s -8s %(message)s"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
