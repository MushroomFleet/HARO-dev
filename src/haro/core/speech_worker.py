"""Speech output worker with priority queue.

Handles TTS synthesis and audio playback with priority-based queuing.
Status phrases get higher priority than responses.
"""

import asyncio
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Callable, Any

import numpy as np

from haro.audio.playback import AudioPlayback
from haro.speech.tts import PiperTTS, SynthesisResult
from haro.utils.logging import get_logger

logger = get_logger(__name__)


class SpeechPriority(IntEnum):
    """Priority levels for speech output.

    Note: Lower numbers = higher priority. However, to ensure proper
    sequencing of conversational flow, STATUS and RESPONSE now share
    the same priority (2) so they play in FIFO order. Only URGENT (0)
    interrupts the queue.
    """

    URGENT = 0  # Immediate status (errors, interrupts) - jumps queue
    STATUS = 2  # Normal status (processing, thinking) - FIFO with responses
    RESPONSE = 2  # Full API responses - FIFO with status


@dataclass(order=True)
class SpeechItem:
    """Item in the speech queue."""

    priority: int
    text: str = field(compare=False)
    callback: Optional[Callable[[], Any]] = field(default=None, compare=False)
    signoff: Optional[str] = field(default=None, compare=False)
    sequence: int = field(default=0, compare=True)  # For FIFO within same priority


class SpeechWorker:
    """Priority-based speech output worker.

    Processes speech requests from a priority queue, synthesizing text
    and playing audio. Higher priority items (lower numbers) are processed
    first, enabling status phrases to interrupt queued responses.
    """

    def __init__(
        self,
        tts: PiperTTS,
        playback: AudioPlayback,
        signoff: str = "HARO",
    ) -> None:
        """Initialize speech worker.

        Args:
            tts: TTS engine for synthesis.
            playback: Audio playback for output.
            signoff: Default signoff phrase.
        """
        self._tts = tts
        self._playback = playback
        self._signoff = signoff

        self._queue: asyncio.PriorityQueue[SpeechItem] = asyncio.PriorityQueue()
        self._running = False
        self._sequence = 0  # Monotonic counter for FIFO ordering
        self._current_item: Optional[SpeechItem] = None
        self._task: Optional[asyncio.Task] = None
        self._interrupted = False  # Flag to signal interrupt occurred

        self.logger = logger.bind(component="SpeechWorker")

    async def start(self) -> None:
        """Start the speech worker."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run())
        self.logger.info("speech_worker_started")

    async def stop(self) -> None:
        """Stop the speech worker."""
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        # Clear queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        self.logger.info("speech_worker_stopped")

    async def speak(
        self,
        text: str,
        priority: SpeechPriority = SpeechPriority.RESPONSE,
        signoff: Optional[str] = None,
        callback: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Queue text for speech output.

        Args:
            text: Text to speak.
            priority: Priority level (lower = higher priority).
            signoff: Optional signoff to append.
            callback: Optional callback after speech completes.
        """
        self._sequence += 1
        item = SpeechItem(
            priority=priority,
            text=text,
            signoff=signoff,
            callback=callback,
            sequence=self._sequence,
        )
        await self._queue.put(item)
        self.logger.debug(
            "speech_queued",
            text=text[:50],
            priority=priority.name if isinstance(priority, SpeechPriority) else priority,
            queue_size=self._queue.qsize(),
        )

    async def speak_status(self, text: str, urgent: bool = False) -> None:
        """Speak a status phrase with high priority.

        Args:
            text: Status text to speak.
            urgent: If True, use URGENT priority (interrupts everything).
        """
        priority = SpeechPriority.URGENT if urgent else SpeechPriority.STATUS
        await self.speak(text, priority=priority, signoff=self._signoff)

    async def speak_response(
        self,
        text: str,
        double_signoff: bool = True,
        callback: Optional[Callable[[], Any]] = None,
    ) -> None:
        """Speak an API response with normal priority.

        Args:
            text: Response text to speak.
            double_signoff: If True, use double signoff (HARO HARO).
            callback: Optional callback after speech completes.
        """
        signoff = f"{self._signoff} {self._signoff}" if double_signoff else self._signoff
        await self.speak(
            text,
            priority=SpeechPriority.RESPONSE,
            signoff=signoff,
            callback=callback,
        )

    async def interrupt(self) -> None:
        """Interrupt current speech and clear queue.

        This stops the current playback immediately without stopping
        the playback system, allowing new speech to be queued.
        """
        # Set interrupted flag so wait_for_speech returns immediately
        self._interrupted = True

        # Interrupt current playback (doesn't stop the playback loop)
        if self._playback:
            self._playback.interrupt()
            self._playback.clear_queue()

        # Clear speech queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Clear current item so we don't wait for it
        self._current_item = None

        self.logger.info("speech_interrupted")

    @property
    def is_speaking(self) -> bool:
        """Check if currently speaking."""
        return self._current_item is not None or self._playback.is_playing

    @property
    def queue_size(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    async def wait_for_speech(self, timeout: Optional[float] = None) -> bool:
        """Wait for current speech and queue to complete.

        Args:
            timeout: Optional timeout in seconds.

        Returns:
            True if speech completed, False if timed out or interrupted.
        """
        start_time = asyncio.get_event_loop().time()

        while self.is_speaking or self.queue_size > 0:
            # Check if we were interrupted - return immediately
            if self._interrupted:
                self._interrupted = False  # Reset flag
                return False

            if timeout:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed >= timeout:
                    return False
            await asyncio.sleep(0.1)

        return True

    async def _run(self) -> None:
        """Main worker loop."""
        self.logger.info("speech_worker_running")

        while self._running:
            try:
                # Wait for item with timeout
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

                self._current_item = item

                # Build full text with signoff
                full_text = item.text
                if item.signoff:
                    full_text = f"{item.text} {item.signoff}"

                self.logger.debug(
                    "speaking",
                    text=full_text[:50],
                    priority=item.priority,
                )

                # Synthesize and play
                try:
                    result = await self._tts.synthesize(full_text)
                    await self._playback.play(
                        result.audio,
                        sample_rate=result.sample_rate,
                        wait=True,
                    )

                    # Execute callback if provided
                    if item.callback:
                        try:
                            if asyncio.iscoroutinefunction(item.callback):
                                await item.callback()
                            else:
                                item.callback()
                        except Exception as e:
                            self.logger.error("callback_error", error=str(e))

                except Exception as e:
                    self.logger.error("speech_error", error=str(e), text=item.text[:50])

                finally:
                    self._current_item = None
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error("worker_error", error=str(e))

        self.logger.info("speech_worker_loop_exited")
