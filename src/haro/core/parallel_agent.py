"""Parallel HARO agent using worker architecture.

Replaces the sequential state machine with concurrent workers for
reduced latency and immediate speech feedback.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from haro.core.config import HaroConfig
from haro.core.events import EventBus
from haro.core.listen_worker import ListenWorker
from haro.core.speech_worker import SpeechWorker
from haro.core.orchestrator import Orchestrator
from haro.audio.capture import AudioCapture
from haro.audio.playback import AudioPlayback
from haro.audio.vad import VoiceActivityDetector
from haro.audio.wake import WakeWordDetector
from haro.speech.stt import WhisperSTT
from haro.speech.tts import PiperTTS
from haro.intelligence.client import ClaudeClient
from haro.context.manager import ContextManager
from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParallelAgentStats:
    """Runtime statistics for parallel agent."""

    start_time: float = field(default_factory=time.time)
    requests: int = 0
    errors: int = 0

    @property
    def uptime(self) -> float:
        """Get agent uptime in seconds."""
        return time.time() - self.start_time


class ParallelAgent:
    """Parallel HARO agent using worker architecture.

    Uses three concurrent workers:
    - ListenWorker: Continuous audio capture, wake detection, VAD, STT
    - SpeechWorker: Priority-based TTS synthesis and playback
    - Orchestrator: Coordinates workers, handles API calls
    """

    def __init__(
        self,
        config: HaroConfig,
        event_bus: Optional[EventBus] = None,
    ) -> None:
        """Initialize parallel agent.

        Args:
            config: HARO configuration.
            event_bus: Optional event bus.
        """
        self.config = config
        self.event_bus = event_bus or EventBus()
        self.stats = ParallelAgentStats()

        # Workers (set during initialize)
        self._listen_worker: Optional[ListenWorker] = None
        self._speech_worker: Optional[SpeechWorker] = None
        self._orchestrator: Optional[Orchestrator] = None

        # Components (set during initialize)
        self._audio_capture: Optional[AudioCapture] = None
        self._audio_playback: Optional[AudioPlayback] = None
        self._vad: Optional[VoiceActivityDetector] = None
        self._wake_detector: Optional[WakeWordDetector] = None
        self._stt: Optional[WhisperSTT] = None
        self._tts: Optional[PiperTTS] = None
        self._api_client: Optional[ClaudeClient] = None
        self._context_manager: Optional[ContextManager] = None

        # Control
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._tasks: list[asyncio.Task] = []

        self.logger = logger.bind(component="ParallelAgent")

    @property
    def is_running(self) -> bool:
        """Check if agent is running."""
        return self._running

    async def initialize(
        self,
        audio_capture: AudioCapture,
        audio_playback: AudioPlayback,
        wake_detector: WakeWordDetector,
        vad: VoiceActivityDetector,
        stt: WhisperSTT,
        tts: Optional[PiperTTS],
        api_client: Optional[ClaudeClient],
        context_manager: Optional[ContextManager] = None,
        prompt_builder=None,
        **kwargs,  # Accept additional args for compatibility
    ) -> None:
        """Initialize the agent with component dependencies.

        Args:
            audio_capture: Audio capture instance.
            audio_playback: Audio playback instance.
            wake_detector: Wake word detector.
            vad: Voice activity detector.
            stt: Speech-to-text engine.
            tts: Text-to-speech engine.
            api_client: Claude API client.
            context_manager: Optional context manager.
            prompt_builder: Optional prompt builder for system prompts.
        """
        self._audio_capture = audio_capture
        self._audio_playback = audio_playback
        self._vad = vad
        self._wake_detector = wake_detector
        self._stt = stt
        self._tts = tts
        self._api_client = api_client
        self._context_manager = context_manager
        self._prompt_builder = prompt_builder

        # Initialize wake detector with STT
        if self._wake_detector and self._stt:
            await self._wake_detector.initialize(self._stt)

        # Initialize API client
        if self._api_client and hasattr(self._api_client, 'initialize'):
            await self._api_client.initialize()

        # Create workers
        if tts:
            self._speech_worker = SpeechWorker(
                tts=tts,
                playback=audio_playback,
                signoff=self.config.wake.phrase,
            )

        self._listen_worker = ListenWorker(
            capture=audio_capture,
            vad=vad,
            wake_detector=wake_detector,
            stt=stt,
            sample_rate=self.config.audio.sample_rate,
        )

        self._orchestrator = Orchestrator(
            listen_worker=self._listen_worker,
            speech_worker=self._speech_worker,
            api_client=api_client,
            config=self.config,
            context_manager=context_manager,
            prompt_builder=prompt_builder,
        )

        self.logger.info(
            "parallel_agent_initialized",
            has_capture=audio_capture is not None,
            has_playback=audio_playback is not None,
            has_wake=wake_detector is not None,
            has_stt=stt is not None,
            has_tts=tts is not None,
            has_api=api_client is not None,
        )

    async def run(self) -> None:
        """Run the parallel agent.

        Starts all workers concurrently and runs until shutdown.
        """
        if self._running:
            self.logger.warning("agent_already_running")
            return

        self._running = True
        self._shutdown_event.clear()
        self.stats = ParallelAgentStats()

        self.logger.info("parallel_agent_starting")

        # Start audio capture
        if self._audio_capture:
            await self._audio_capture.start()
            self.logger.info("audio_capture_started")

        # Start audio playback
        if self._audio_playback:
            await self._audio_playback.start()
            self.logger.info("audio_playback_started")

        try:
            # Start all workers
            worker_tasks = []

            if self._listen_worker:
                await self._listen_worker.start()
                worker_tasks.append(
                    asyncio.create_task(
                        self._listen_worker._run(),
                        name="listen_worker"
                    )
                )

            if self._speech_worker:
                await self._speech_worker.start()
                worker_tasks.append(
                    asyncio.create_task(
                        self._speech_worker._run(),
                        name="speech_worker"
                    )
                )

            if self._orchestrator:
                await self._orchestrator.start()
                worker_tasks.append(
                    asyncio.create_task(
                        self._orchestrator._run(),
                        name="orchestrator"
                    )
                )

            self._tasks = worker_tasks
            self.logger.info(
                "workers_started",
                count=len(worker_tasks),
            )

            # Wait for shutdown or worker failure
            shutdown_task = asyncio.create_task(
                self._shutdown_event.wait(),
                name="shutdown_wait"
            )

            done, pending = await asyncio.wait(
                worker_tasks + [shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Check if a worker crashed
            for task in done:
                if task.get_name() != "shutdown_wait":
                    try:
                        task.result()
                    except Exception as e:
                        self.logger.error(
                            "worker_crashed",
                            worker=task.get_name(),
                            error=str(e),
                        )

            # Cancel remaining tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            self.logger.info("agent_cancelled")

        finally:
            self._running = False
            await self._cleanup()
            self.logger.info(
                "parallel_agent_stopped",
                uptime=self.stats.uptime,
            )

    async def stop(self) -> None:
        """Request graceful shutdown."""
        self.logger.info("agent_stop_requested")
        self._running = False
        self._shutdown_event.set()

        # Stop workers
        if self._orchestrator:
            await self._orchestrator.stop()
        if self._speech_worker:
            await self._speech_worker.stop()
        if self._listen_worker:
            await self._listen_worker.stop()

    async def _cleanup(self) -> None:
        """Clean up resources."""
        # Stop workers
        if self._orchestrator:
            await self._orchestrator.stop()
        if self._speech_worker:
            await self._speech_worker.stop()
        if self._listen_worker:
            await self._listen_worker.stop()

        # Stop audio
        if self._audio_capture:
            await self._audio_capture.stop()
        if self._audio_playback:
            await self._audio_playback.stop()

        self.logger.info("cleanup_complete")
