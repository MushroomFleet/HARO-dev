"""HARO audio module.

Provides audio capture, playback, voice activity detection, and wake word detection.
"""

from haro.audio.capture import AudioCapture
from haro.audio.playback import AudioPlayback
from haro.audio.vad import VoiceActivityDetector, VADState, VADResult
from haro.audio.wake import (
    WakeWordDetector,
    WakeWordResult,
    RollingBuffer,
    ContinuousWakeDetector,
)
from haro.audio.feedback import AudioFeedback, FeedbackConfig

__all__ = [
    "AudioCapture",
    "AudioPlayback",
    "VoiceActivityDetector",
    "VADState",
    "VADResult",
    "WakeWordDetector",
    "WakeWordResult",
    "RollingBuffer",
    "ContinuousWakeDetector",
    "AudioFeedback",
    "FeedbackConfig",
]
