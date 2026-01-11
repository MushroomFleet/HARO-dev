"""Performance profiling utilities for HARO.

Provides timing decorators and profiling tools to measure
component performance and identify bottlenecks.
"""

import asyncio
import functools
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional, Callable, Any
from collections import defaultdict

from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TimingStats:
    """Statistics for a timed operation."""

    name: str
    total_time: float = 0.0
    call_count: int = 0
    min_time: float = float("inf")
    max_time: float = 0.0
    last_time: float = 0.0

    @property
    def avg_time(self) -> float:
        """Get average execution time."""
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count

    def record(self, duration: float) -> None:
        """Record a timing measurement.

        Args:
            duration: Duration in seconds.
        """
        self.total_time += duration
        self.call_count += 1
        self.last_time = duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)

    def to_dict(self) -> dict:
        """Convert to dictionary.

        Returns:
            Dictionary of statistics.
        """
        return {
            "name": self.name,
            "total_time_ms": self.total_time * 1000,
            "call_count": self.call_count,
            "avg_time_ms": self.avg_time * 1000,
            "min_time_ms": self.min_time * 1000 if self.min_time != float("inf") else 0,
            "max_time_ms": self.max_time * 1000,
            "last_time_ms": self.last_time * 1000,
        }


class Profiler:
    """Performance profiler for HARO components.

    Tracks timing statistics across the application and provides
    reporting capabilities.
    """

    _instance: Optional["Profiler"] = None

    def __init__(self) -> None:
        """Initialize profiler."""
        self._stats: dict[str, TimingStats] = {}
        self._enabled = True
        self._start_time = time.time()
        self.logger = logger.bind(component="Profiler")

    @classmethod
    def get_instance(cls) -> "Profiler":
        """Get singleton profiler instance.

        Returns:
            The global Profiler instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def enabled(self) -> bool:
        """Check if profiling is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Set profiling enabled state."""
        self._enabled = value

    def record(self, name: str, duration: float) -> None:
        """Record a timing measurement.

        Args:
            name: Name of the operation.
            duration: Duration in seconds.
        """
        if not self._enabled:
            return

        if name not in self._stats:
            self._stats[name] = TimingStats(name=name)

        self._stats[name].record(duration)

    @contextmanager
    def time_block(self, name: str, log: bool = False):
        """Context manager for timing a code block.

        Args:
            name: Name for the timed operation.
            log: Whether to log the timing.

        Yields:
            None
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.record(name, duration)

            if log:
                self.logger.debug(
                    "timing",
                    operation=name,
                    duration_ms=f"{duration * 1000:.2f}",
                )

    def get_stats(self, name: Optional[str] = None) -> dict:
        """Get profiling statistics.

        Args:
            name: Specific operation name, or None for all.

        Returns:
            Dictionary of statistics.
        """
        if name:
            if name in self._stats:
                return self._stats[name].to_dict()
            return {}

        return {
            "uptime_seconds": time.time() - self._start_time,
            "operations": {
                name: stats.to_dict()
                for name, stats in self._stats.items()
            },
        }

    def get_summary(self) -> str:
        """Get human-readable summary.

        Returns:
            Formatted summary string.
        """
        lines = ["Performance Summary", "=" * 40]

        if not self._stats:
            lines.append("No timing data recorded.")
            return "\n".join(lines)

        # Sort by total time
        sorted_stats = sorted(
            self._stats.values(),
            key=lambda s: s.total_time,
            reverse=True,
        )

        for stats in sorted_stats:
            lines.append(
                f"{stats.name}: "
                f"calls={stats.call_count}, "
                f"avg={stats.avg_time * 1000:.2f}ms, "
                f"total={stats.total_time * 1000:.2f}ms"
            )

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset all statistics."""
        self._stats.clear()
        self._start_time = time.time()


def timed(name: Optional[str] = None, log: bool = False):
    """Decorator for timing synchronous functions.

    Args:
        name: Name for the operation (defaults to function name).
        log: Whether to log timing.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        op_name = name or func.__qualname__

        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            profiler = Profiler.get_instance()
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                profiler.record(op_name, duration)
                if log:
                    logger.debug(
                        "timing",
                        operation=op_name,
                        duration_ms=f"{duration * 1000:.2f}",
                    )

        return wrapper
    return decorator


def async_timed(name: Optional[str] = None, log: bool = False):
    """Decorator for timing async functions.

    Args:
        name: Name for the operation (defaults to function name).
        log: Whether to log timing.

    Returns:
        Decorated function.
    """
    def decorator(func: Callable) -> Callable:
        op_name = name or func.__qualname__

        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            profiler = Profiler.get_instance()
            start = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                duration = time.perf_counter() - start
                profiler.record(op_name, duration)
                if log:
                    logger.debug(
                        "timing",
                        operation=op_name,
                        duration_ms=f"{duration * 1000:.2f}",
                    )

        return wrapper
    return decorator


@dataclass
class LatencyTracker:
    """Track end-to-end latency for voice interactions.

    Measures time from wake word detection to response completion.
    """

    name: str = "interaction"
    start_time: float = 0.0
    checkpoints: dict = field(default_factory=dict)

    def start(self) -> None:
        """Start tracking a new interaction."""
        self.start_time = time.perf_counter()
        self.checkpoints.clear()

    def checkpoint(self, name: str) -> float:
        """Record a checkpoint.

        Args:
            name: Checkpoint name.

        Returns:
            Time since start in seconds.
        """
        elapsed = time.perf_counter() - self.start_time
        self.checkpoints[name] = elapsed
        return elapsed

    def end(self) -> float:
        """End tracking and return total latency.

        Returns:
            Total latency in seconds.
        """
        return time.perf_counter() - self.start_time

    def get_breakdown(self) -> dict:
        """Get latency breakdown by checkpoint.

        Returns:
            Dictionary of checkpoint timings.
        """
        if not self.checkpoints:
            return {}

        result = {}
        prev_time = 0.0
        sorted_checkpoints = sorted(
            self.checkpoints.items(),
            key=lambda x: x[1],
        )

        for name, timestamp in sorted_checkpoints:
            result[name] = {
                "elapsed_ms": timestamp * 1000,
                "delta_ms": (timestamp - prev_time) * 1000,
            }
            prev_time = timestamp

        return result


class MemoryTracker:
    """Track memory usage.

    Note: Requires psutil for accurate tracking.
    """

    def __init__(self) -> None:
        """Initialize memory tracker."""
        self._baseline: Optional[int] = None
        self._process = None
        self._available = False

        try:
            import psutil
            self._process = psutil.Process()
            self._available = True
        except ImportError:
            pass

    @property
    def available(self) -> bool:
        """Check if memory tracking is available."""
        return self._available

    def get_current_mb(self) -> float:
        """Get current memory usage in MB.

        Returns:
            Memory usage in megabytes.
        """
        if not self._available:
            return 0.0

        info = self._process.memory_info()
        return info.rss / (1024 * 1024)

    def set_baseline(self) -> None:
        """Set current memory as baseline."""
        if not self._available:
            return

        info = self._process.memory_info()
        self._baseline = info.rss

    def get_delta_mb(self) -> float:
        """Get memory change from baseline in MB.

        Returns:
            Memory delta in megabytes.
        """
        if not self._available or self._baseline is None:
            return 0.0

        info = self._process.memory_info()
        return (info.rss - self._baseline) / (1024 * 1024)

    def get_summary(self) -> dict:
        """Get memory summary.

        Returns:
            Dictionary of memory statistics.
        """
        if not self._available:
            return {"available": False}

        info = self._process.memory_info()
        return {
            "available": True,
            "rss_mb": info.rss / (1024 * 1024),
            "vms_mb": info.vms / (1024 * 1024),
            "baseline_mb": (self._baseline / (1024 * 1024)) if self._baseline else None,
        }


# Global profiler access
def get_profiler() -> Profiler:
    """Get the global profiler instance.

    Returns:
        The Profiler singleton.
    """
    return Profiler.get_instance()
