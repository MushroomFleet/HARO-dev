"""HARO speech module.

Provides speech-to-text (STT) and text-to-speech (TTS) capabilities
using Whisper and Piper respectively.
"""

from haro.speech.stt import WhisperSTT, TranscriptionResult
from haro.speech.tts import PiperTTS, SynthesisResult
from haro.speech.models import ModelManager, ModelInfo, DownloadProgress

__all__ = [
    "WhisperSTT",
    "TranscriptionResult",
    "PiperTTS",
    "SynthesisResult",
    "ModelManager",
    "ModelInfo",
    "DownloadProgress",
]
