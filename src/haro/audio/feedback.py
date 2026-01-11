"""Audio feedback for HARO.

Provides audio confirmation sounds and verbal feedback for wake word
detection and other events.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import Optional, List

import numpy as np

from haro.core.config import WakeConfig, TTSConfig
from haro.audio.playback import AudioPlayback
from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeedbackConfig:
    """Configuration for audio feedback."""

    confirmation_sound: bool = True
    confirmation_phrases: List[str] = None
    chime_frequency: float = 880.0  # A5
    chime_duration: float = 0.15
    chime_volume: float = 0.5

    def __post_init__(self):
        if self.confirmation_phrases is None:
            self.confirmation_phrases = [
                "Yes?",
                "I'm here.",
                "Listening.",
                "How can I help?",
            ]


class AudioFeedback:
    """Audio feedback for HARO events.

    Provides confirmation sounds and verbal feedback using either
    simple tones or TTS synthesis.
    """

    def __init__(
        self,
        playback: AudioPlayback,
        wake_config: Optional[WakeConfig] = None,
        tts_config: Optional[TTSConfig] = None,
    ) -> None:
        """Initialize audio feedback.

        Args:
            playback: AudioPlayback instance for playing sounds.
            wake_config: Optional wake configuration for phrases.
            tts_config: Optional TTS configuration for verbal feedback.
        """
        self.playback = playback
        self._tts = None
        self._tts_config = tts_config

        # Build feedback config from wake config
        if wake_config:
            self.config = FeedbackConfig(
                confirmation_sound=wake_config.confirmation_sound,
                confirmation_phrases=wake_config.confirmation_phrases,
            )
        else:
            self.config = FeedbackConfig()

        # Pre-generate chime
        self._chime = self._generate_chime()

        self.logger = logger.bind(component="AudioFeedback")

    def _generate_chime(self) -> np.ndarray:
        """Generate a confirmation chime sound.

        Creates a pleasant two-tone chime using sine waves.

        Returns:
            Audio samples for chime.
        """
        sample_rate = 16000
        duration = self.config.chime_duration
        volume = self.config.chime_volume

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # Two-tone chime (A5 -> E6)
        freq1 = self.config.chime_frequency  # 880 Hz
        freq2 = freq1 * 1.5  # 1320 Hz (perfect fifth)

        # First tone
        tone1 = np.sin(2 * np.pi * freq1 * t[:len(t)//2])
        # Second tone
        tone2 = np.sin(2 * np.pi * freq2 * t[len(t)//2:])

        # Combine
        audio = np.concatenate([tone1, tone2]) * volume

        # Apply envelope (fade in/out)
        fade_samples = int(sample_rate * 0.02)  # 20ms fade
        if len(audio) > fade_samples * 2:
            # Fade in
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            # Fade out
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        return audio.astype(np.float32)

    def _generate_error_sound(self) -> np.ndarray:
        """Generate an error/failure sound.

        Returns:
            Audio samples for error sound.
        """
        sample_rate = 16000
        duration = 0.3
        volume = 0.4

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # Descending tone (error indication)
        freq_start = 600
        freq_end = 300
        freq = np.linspace(freq_start, freq_end, len(t))
        audio = np.sin(2 * np.pi * freq * t / sample_rate * np.arange(len(t))) * volume

        # Apply envelope
        fade_samples = int(sample_rate * 0.02)
        if len(audio) > fade_samples * 2:
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)

        return audio.astype(np.float32)

    async def set_tts(self, tts) -> None:
        """Set TTS instance for verbal feedback.

        Args:
            tts: PiperTTS instance for speech synthesis.
        """
        self._tts = tts
        self.logger.debug("tts_configured")

    async def play_wake_confirmation(self, use_verbal: bool = False) -> None:
        """Play wake word confirmation feedback.

        Args:
            use_verbal: If True, use verbal phrase instead of chime.
        """
        if not self.config.confirmation_sound:
            return

        if use_verbal and self._tts and self.config.confirmation_phrases:
            await self._play_verbal_confirmation()
        else:
            await self._play_chime()

    async def _play_chime(self) -> None:
        """Play the confirmation chime."""
        try:
            await self.playback.play(
                self._chime,
                sample_rate=16000,
                wait=True,
            )
            self.logger.debug("chime_played")
        except Exception as e:
            self.logger.error("chime_failed", error=str(e))

    async def _play_verbal_confirmation(self) -> None:
        """Play a verbal confirmation phrase."""
        if not self._tts or not self.config.confirmation_phrases:
            await self._play_chime()
            return

        phrase = random.choice(self.config.confirmation_phrases)

        try:
            result = await self._tts.synthesize(phrase)
            await self.playback.play(
                result.audio,
                sample_rate=result.sample_rate,
                wait=True,
            )
            self.logger.debug("verbal_confirmation_played", phrase=phrase)
        except Exception as e:
            self.logger.error("verbal_confirmation_failed", error=str(e))
            # Fall back to chime
            await self._play_chime()

    async def play_error(self) -> None:
        """Play error indication sound."""
        try:
            error_sound = self._generate_error_sound()
            await self.playback.play(
                error_sound,
                sample_rate=16000,
                wait=True,
            )
            self.logger.debug("error_sound_played")
        except Exception as e:
            self.logger.error("error_sound_failed", error=str(e))

    async def play_processing(self) -> None:
        """Play processing indication sound.

        A subtle sound to indicate HARO is processing.
        """
        # Short, soft beep
        sample_rate = 16000
        duration = 0.1
        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)
        audio = np.sin(2 * np.pi * 660 * t) * 0.2  # E5, soft

        # Fade
        fade = int(sample_rate * 0.01)
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)

        try:
            await self.playback.play(audio, sample_rate=16000, wait=True)
        except Exception:
            pass

    async def play_ready(self) -> None:
        """Play ready/startup sound.

        An ascending tone to indicate HARO is ready.
        """
        sample_rate = 16000
        duration = 0.4
        volume = 0.4

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # Three ascending tones (C5 -> E5 -> G5)
        freqs = [523.25, 659.25, 783.99]  # C5, E5, G5
        segment_len = len(t) // 3

        audio = np.zeros_like(t)
        for i, freq in enumerate(freqs):
            start = i * segment_len
            end = start + segment_len
            segment_t = t[:segment_len]
            audio[start:end] = np.sin(2 * np.pi * freq * segment_t)

        audio *= volume

        # Envelope
        fade = int(sample_rate * 0.02)
        audio[:fade] *= np.linspace(0, 1, fade)
        audio[-fade:] *= np.linspace(1, 0, fade)

        try:
            await self.playback.play(audio.astype(np.float32), sample_rate=16000, wait=True)
            self.logger.debug("ready_sound_played")
        except Exception as e:
            self.logger.error("ready_sound_failed", error=str(e))

    async def play_goodbye(self) -> None:
        """Play goodbye/shutdown sound.

        A descending tone to indicate HARO is shutting down.
        """
        sample_rate = 16000
        duration = 0.5
        volume = 0.3

        t = np.linspace(0, duration, int(sample_rate * duration), dtype=np.float32)

        # Three descending tones (G5 -> E5 -> C5)
        freqs = [783.99, 659.25, 523.25]  # G5, E5, C5
        segment_len = len(t) // 3

        audio = np.zeros_like(t)
        for i, freq in enumerate(freqs):
            start = i * segment_len
            end = start + segment_len
            segment_t = t[:segment_len]
            audio[start:end] = np.sin(2 * np.pi * freq * segment_t)

        audio *= volume

        # Envelope with longer fade out
        fade_in = int(sample_rate * 0.02)
        fade_out = int(sample_rate * 0.1)
        audio[:fade_in] *= np.linspace(0, 1, fade_in)
        audio[-fade_out:] *= np.linspace(1, 0, fade_out)

        try:
            await self.playback.play(audio.astype(np.float32), sample_rate=16000, wait=True)
            self.logger.debug("goodbye_sound_played")
        except Exception as e:
            self.logger.error("goodbye_sound_failed", error=str(e))

    async def play_acknowledgment(self, query: str = "", wait: bool = True) -> None:
        """Play immediate acknowledgment after receiving a query.

        Selects a context-aware phrase based on query keywords.

        Args:
            query: The user's query for context-aware phrase selection.
            wait: If True, wait for playback to complete.
        """
        if not self._tts:
            # Fall back to processing beep if no TTS
            await self.play_processing()
            return

        # Context-aware phrase selection
        query_lower = query.lower()

        context_phrases = {
            "weather": ["Checking the weather", "Let me check the weather"],
            "time": ["Getting the time", "Let me check"],
            "date": ["Checking the date", "Let me see"],
            "remind": ["I'll set that reminder", "Setting a reminder"],
            "search": ["Searching for that", "Let me search"],
            "calculate": ["Calculating", "Let me work that out"],
            "tell me": ["Let me tell you", "I can help with that"],
            "what is": ["Let me look that up", "I'll find out"],
            "who is": ["Let me check", "Looking that up"],
            "how": ["Let me explain", "I'll help with that"],
        }

        # Find matching context phrase
        phrase = None
        for keyword, phrases in context_phrases.items():
            if keyword in query_lower:
                phrase = random.choice(phrases)
                break

        # Default phrases if no context match
        if not phrase:
            default_phrases = [
                "I'll look that up",
                "Let me check",
                "One moment",
                "Looking into that",
            ]
            phrase = random.choice(default_phrases)

        try:
            result = await self._tts.synthesize(phrase)
            await self.playback.play(
                result.audio,
                sample_rate=result.sample_rate,
                wait=wait,
            )
            self.logger.debug("acknowledgment_played", phrase=phrase)
        except Exception as e:
            self.logger.error("acknowledgment_failed", error=str(e))
            # Fall back to beep
            await self.play_processing()

    async def play_thinking(self, wait: bool = True) -> None:
        """Play a 'thinking/processing' phrase while waiting for API.

        Args:
            wait: If True, wait for playback to complete.
        """
        if not self._tts:
            await self.play_processing()
            return

        phrases = [
            "Getting the info now",
            "Working on it",
            "Just a moment",
            "Almost there",
        ]
        phrase = random.choice(phrases)

        try:
            result = await self._tts.synthesize(phrase)
            await self.playback.play(
                result.audio,
                sample_rate=result.sample_rate,
                wait=wait,
            )
            self.logger.debug("thinking_phrase_played", phrase=phrase)
        except Exception as e:
            self.logger.error("thinking_phrase_failed", error=str(e))

    def add_signoff(self, response: str, signoff: str = "HARO") -> str:
        """Append a sign-off word to a response.

        Args:
            response: The response text.
            signoff: The sign-off word to append.

        Returns:
            Response with sign-off appended.
        """
        # Clean up response ending
        response = response.rstrip()

        # Don't add signoff if response already ends with it
        if response.lower().endswith(signoff.lower()):
            return response

        # Add appropriate punctuation before signoff
        if response and response[-1] in ".!?":
            return f"{response} {signoff}."
        elif response:
            return f"{response}, {signoff}."
        else:
            return response
