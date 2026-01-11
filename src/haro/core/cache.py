"""Response caching for HARO.

Implements LRU cache with TTL for API responses to reduce
latency and API costs for repeated questions.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict

from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CacheEntry:
    """A cached response entry."""

    response: str
    created_at: float
    hits: int = 0
    last_accessed: float = field(default_factory=time.time)

    @property
    def age(self) -> float:
        """Get age of entry in seconds."""
        return time.time() - self.created_at


@dataclass
class CacheConfig:
    """Response cache configuration."""

    enabled: bool = True
    max_entries: int = 100
    ttl_seconds: float = 3600.0  # 1 hour default
    similarity_threshold: float = 0.85


class ResponseCache:
    """LRU cache for API responses.

    Features:
    - LRU eviction when max_entries exceeded
    - TTL-based expiration
    - Similarity-based matching for similar questions
    - Statistics tracking
    """

    def __init__(self, config: Optional[CacheConfig] = None) -> None:
        """Initialize response cache.

        Args:
            config: Cache configuration.
        """
        self.config = config or CacheConfig()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "expirations": 0,
        }
        self.logger = logger.bind(component="ResponseCache")

    @property
    def size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    @property
    def hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self._stats["hits"] + self._stats["misses"]
        if total == 0:
            return 0.0
        return self._stats["hits"] / total

    def get(self, query: str) -> Optional[str]:
        """Get cached response for a query.

        Args:
            query: The user's query.

        Returns:
            Cached response or None if not found.
        """
        if not self.config.enabled:
            return None

        key = self._normalize_key(query)

        # Check exact match first
        entry = self._cache.get(key)
        if entry:
            if entry.age > self.config.ttl_seconds:
                # Entry expired
                self._remove(key)
                self._stats["expirations"] += 1
                self._stats["misses"] += 1
                return None

            # Cache hit - move to end (most recently used)
            self._cache.move_to_end(key)
            entry.hits += 1
            entry.last_accessed = time.time()
            self._stats["hits"] += 1

            self.logger.debug(
                "cache_hit",
                key=key[:30],
                hits=entry.hits,
            )
            return entry.response

        # Try fuzzy matching
        similar_key = self._find_similar(query)
        if similar_key:
            entry = self._cache[similar_key]
            if entry.age <= self.config.ttl_seconds:
                self._cache.move_to_end(similar_key)
                entry.hits += 1
                entry.last_accessed = time.time()
                self._stats["hits"] += 1

                self.logger.debug(
                    "cache_hit_similar",
                    original=query[:30],
                    matched=similar_key[:30],
                )
                return entry.response

        self._stats["misses"] += 1
        return None

    def put(self, query: str, response: str) -> None:
        """Store a response in cache.

        Args:
            query: The user's query.
            response: The API response.
        """
        if not self.config.enabled:
            return

        key = self._normalize_key(query)

        # Update existing or create new
        if key in self._cache:
            self._cache[key].response = response
            self._cache[key].created_at = time.time()
            self._cache.move_to_end(key)
        else:
            # Evict if at capacity
            while len(self._cache) >= self.config.max_entries:
                self._evict_oldest()

            self._cache[key] = CacheEntry(
                response=response,
                created_at=time.time(),
            )

        self.logger.debug(
            "cache_put",
            key=key[:30],
            size=len(self._cache),
        )

    def invalidate(self, query: Optional[str] = None) -> int:
        """Invalidate cache entries.

        Args:
            query: Specific query to invalidate, or None for all.

        Returns:
            Number of entries invalidated.
        """
        if query is None:
            count = len(self._cache)
            self._cache.clear()
            self.logger.info("cache_cleared", count=count)
            return count

        key = self._normalize_key(query)
        if key in self._cache:
            self._remove(key)
            return 1
        return 0

    def cleanup_expired(self) -> int:
        """Remove expired entries.

        Returns:
            Number of entries removed.
        """
        now = time.time()
        expired = [
            key for key, entry in self._cache.items()
            if now - entry.created_at > self.config.ttl_seconds
        ]

        for key in expired:
            self._remove(key)
            self._stats["expirations"] += 1

        if expired:
            self.logger.debug("cache_cleanup", removed=len(expired))

        return len(expired)

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            Dictionary of cache statistics.
        """
        return {
            "size": self.size,
            "max_entries": self.config.max_entries,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "hit_rate": self.hit_rate,
            "evictions": self._stats["evictions"],
            "expirations": self._stats["expirations"],
            "ttl_seconds": self.config.ttl_seconds,
        }

    def _normalize_key(self, query: str) -> str:
        """Normalize query to cache key.

        Args:
            query: The user's query.

        Returns:
            Normalized cache key.
        """
        # Lowercase, strip whitespace, remove punctuation
        normalized = query.lower().strip()
        # Remove common filler words for better matching
        fillers = ["please", "can you", "could you", "would you", "tell me"]
        for filler in fillers:
            normalized = normalized.replace(filler, "")
        normalized = " ".join(normalized.split())  # Normalize whitespace
        return normalized

    def _find_similar(self, query: str) -> Optional[str]:
        """Find similar cached query.

        Uses simple word overlap for similarity.

        Args:
            query: The user's query.

        Returns:
            Similar cache key or None.
        """
        if not self._cache:
            return None

        query_words = set(self._normalize_key(query).split())
        if not query_words:
            return None

        best_match = None
        best_score = 0.0

        for key in self._cache:
            key_words = set(key.split())
            if not key_words:
                continue

            # Jaccard similarity
            intersection = len(query_words & key_words)
            union = len(query_words | key_words)
            score = intersection / union if union > 0 else 0

            if score > best_score and score >= self.config.similarity_threshold:
                best_score = score
                best_match = key

        return best_match

    def _remove(self, key: str) -> None:
        """Remove entry from cache.

        Args:
            key: Cache key to remove.
        """
        if key in self._cache:
            del self._cache[key]

    def _evict_oldest(self) -> None:
        """Evict the oldest (least recently used) entry."""
        if self._cache:
            key, _ = self._cache.popitem(last=False)
            self._stats["evictions"] += 1
            self.logger.debug("cache_eviction", key=key[:30])
