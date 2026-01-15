"""Text chunking utilities for streaming TTS.

Provides sentence-based chunking for streaming LLM responses,
enabling faster time-to-first-speech by processing sentences
as they arrive rather than waiting for the complete response.
"""

import re
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass
class TextChunk:
    """A chunk of text ready for TTS."""

    text: str
    is_complete: bool  # True if this is a complete sentence
    chunk_index: int


class SentenceChunker:
    """Accumulates streaming text and yields complete sentences.

    Buffers incoming text chunks and emits complete sentences
    as soon as sentence boundaries are detected. This enables
    TTS to start speaking the first sentence while the LLM
    is still generating subsequent sentences.

    Sentence boundaries detected:
    - Period followed by space or end: ". "
    - Question mark followed by space or end: "? "
    - Exclamation mark followed by space or end: "! "
    - Colon followed by space (for "HARO says:"): ": "

    Example:
        chunker = SentenceChunker()
        async for sentence in chunker.process_stream(llm_chunks):
            await tts.synthesize(sentence.text)
    """

    # Sentence-ending punctuation patterns
    # Match period/question/exclamation followed by space or end of string
    SENTENCE_END_PATTERN = re.compile(r'([.!?])\s+')

    # Minimum characters before we consider splitting
    # (prevents splitting on abbreviations like "Dr. Smith")
    MIN_SENTENCE_LENGTH = 20

    def __init__(
        self,
        min_chunk_length: int = 10,
        max_buffer_length: int = 500,
    ) -> None:
        """Initialize the sentence chunker.

        Args:
            min_chunk_length: Minimum characters before yielding a chunk.
            max_buffer_length: Force yield if buffer exceeds this length.
        """
        self._buffer = ""
        self._chunk_index = 0
        self._min_chunk_length = min_chunk_length
        self._max_buffer_length = max_buffer_length

    def reset(self) -> None:
        """Reset the chunker state."""
        self._buffer = ""
        self._chunk_index = 0

    async def process_stream(
        self,
        chunks: AsyncIterator[str],
    ) -> AsyncIterator[TextChunk]:
        """Process a stream of text chunks and yield sentences.

        Args:
            chunks: Async iterator of text chunks from LLM.

        Yields:
            TextChunk objects containing complete sentences.
        """
        self.reset()

        async for chunk in chunks:
            # Add chunk to buffer
            self._buffer += chunk

            # Try to extract complete sentences
            async for sentence in self._extract_sentences():
                yield sentence

        # Yield any remaining buffer content
        if self._buffer.strip():
            yield TextChunk(
                text=self._buffer.strip(),
                is_complete=True,
                chunk_index=self._chunk_index,
            )
            self._chunk_index += 1

    async def _extract_sentences(self) -> AsyncIterator[TextChunk]:
        """Extract complete sentences from the buffer.

        Yields:
            TextChunk objects for each complete sentence found.
        """
        while True:
            # Look for sentence boundaries
            match = self.SENTENCE_END_PATTERN.search(self._buffer)

            if match:
                # Found a sentence boundary
                end_pos = match.end()
                sentence = self._buffer[:end_pos].strip()

                # Only yield if sentence is long enough
                # (helps avoid splitting on abbreviations)
                if len(sentence) >= self._min_chunk_length:
                    yield TextChunk(
                        text=sentence,
                        is_complete=True,
                        chunk_index=self._chunk_index,
                    )
                    self._chunk_index += 1
                    self._buffer = self._buffer[end_pos:]
                else:
                    # Sentence too short, wait for more content
                    break
            elif len(self._buffer) > self._max_buffer_length:
                # Buffer too long without sentence boundary
                # Force split at a reasonable point (comma, semicolon, or space)
                split_pos = self._find_split_point()
                if split_pos > 0:
                    chunk_text = self._buffer[:split_pos].strip()
                    if chunk_text:
                        yield TextChunk(
                            text=chunk_text,
                            is_complete=False,  # Not a complete sentence
                            chunk_index=self._chunk_index,
                        )
                        self._chunk_index += 1
                    self._buffer = self._buffer[split_pos:]
                else:
                    break
            else:
                # No complete sentence yet, wait for more chunks
                break

    def _find_split_point(self) -> int:
        """Find a reasonable point to split a long buffer.

        Returns:
            Position to split at, or 0 if no good split point found.
        """
        # Prefer splitting at punctuation
        for punct in [',', ';', ':']:
            pos = self._buffer.rfind(punct, 0, self._max_buffer_length)
            if pos > self._min_chunk_length:
                return pos + 1

        # Fall back to splitting at last space
        pos = self._buffer.rfind(' ', 0, self._max_buffer_length)
        if pos > self._min_chunk_length:
            return pos + 1

        return 0

    def add_chunk(self, text: str) -> Optional[TextChunk]:
        """Add a text chunk and return a sentence if complete.

        This is a synchronous alternative to process_stream for
        cases where you're processing chunks one at a time.

        Args:
            text: Text chunk to add.

        Returns:
            TextChunk if a complete sentence is ready, None otherwise.
        """
        self._buffer += text

        # Look for sentence boundary
        match = self.SENTENCE_END_PATTERN.search(self._buffer)

        if match and match.end() >= self._min_chunk_length:
            end_pos = match.end()
            sentence = self._buffer[:end_pos].strip()
            self._buffer = self._buffer[end_pos:]

            chunk = TextChunk(
                text=sentence,
                is_complete=True,
                chunk_index=self._chunk_index,
            )
            self._chunk_index += 1
            return chunk

        return None

    def flush(self) -> Optional[TextChunk]:
        """Flush any remaining buffer content.

        Call this when the stream is complete to get any
        remaining text that wasn't emitted as a sentence.

        Returns:
            TextChunk with remaining content, or None if empty.
        """
        if self._buffer.strip():
            chunk = TextChunk(
                text=self._buffer.strip(),
                is_complete=True,
                chunk_index=self._chunk_index,
            )
            self._chunk_index += 1
            self._buffer = ""
            return chunk
        return None
