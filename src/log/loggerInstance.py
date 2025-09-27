from .logger import Logger
from typing import cast
# This file only exists to define the global logger variable

logger: Logger = cast(Logger, None)  # Don't want to initialize the logger without verifying that the log file exists. The 'cast' is a promise that we will assign a Logger later
