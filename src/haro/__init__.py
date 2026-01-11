"""HARO - Helpful Autonomous Responsive Operator.

A lightweight, always-listening voice AI assistant designed for dedicated hardware deployment.
"""

__version__ = "0.1.0"
__author__ = "HARO Project"

from haro.core.config import HaroConfig, load_config
from haro.core.events import EventBus, EventType

__all__ = [
    "__version__",
    "HaroConfig",
    "load_config",
    "EventBus",
    "EventType",
]
