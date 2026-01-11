"""Model download and management for HARO speech components.

Provides utilities for downloading and managing Whisper and Piper models.
"""

import asyncio
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from urllib.parse import urlparse

import httpx

from haro.utils.logging import get_logger

logger = get_logger(__name__)


# Model registry with download URLs and checksums
WHISPER_MODELS = {
    "tiny.en": {
        "url": "https://huggingface.co/Systran/faster-whisper-tiny.en/resolve/main/",
        "files": ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"],
        "size_mb": 75,
    },
    "base.en": {
        "url": "https://huggingface.co/Systran/faster-whisper-base.en/resolve/main/",
        "files": ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"],
        "size_mb": 140,
    },
    "small.en": {
        "url": "https://huggingface.co/Systran/faster-whisper-small.en/resolve/main/",
        "files": ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"],
        "size_mb": 460,
    },
}

PIPER_VOICES = {
    "en_US-lessac-medium": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/",
        "files": ["en_US-lessac-medium.onnx", "en_US-lessac-medium.onnx.json"],
        "size_mb": 60,
    },
    "en_US-amy-medium": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/amy/medium/",
        "files": ["en_US-amy-medium.onnx", "en_US-amy-medium.onnx.json"],
        "size_mb": 60,
    },
    "en_US-ryan-medium": {
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/",
        "files": ["en_US-ryan-medium.onnx", "en_US-ryan-medium.onnx.json"],
        "size_mb": 60,
    },
}


@dataclass
class DownloadProgress:
    """Progress information for downloads."""

    filename: str
    total_bytes: int
    downloaded_bytes: int
    percent: float


@dataclass
class ModelInfo:
    """Information about a model."""

    name: str
    type: str  # "stt" or "tts"
    size_mb: int
    installed: bool
    path: Optional[Path] = None


ProgressCallback = Callable[[DownloadProgress], None]


class ModelManager:
    """Manage speech model downloads and storage."""

    def __init__(
        self,
        base_path: Optional[str] = None,
    ) -> None:
        """Initialize model manager.

        Args:
            base_path: Base path for model storage. Defaults to ~/.cache/haro/models/
        """
        if base_path:
            self.base_path = Path(base_path).expanduser()
        else:
            self.base_path = Path.home() / ".cache" / "haro" / "models"

        self.logger = logger.bind(component="ModelManager")

    def _get_stt_path(self) -> Path:
        """Get path for STT models."""
        return self.base_path / "whisper"

    def _get_tts_path(self) -> Path:
        """Get path for TTS models."""
        return self.base_path / "piper"

    def list_installed_models(self) -> list[ModelInfo]:
        """List all installed models.

        Returns:
            List of ModelInfo for installed models.
        """
        models = []

        # Check Whisper models
        stt_path = self._get_stt_path()
        for model_name in WHISPER_MODELS:
            model_dir = stt_path / model_name
            installed = model_dir.exists() and (model_dir / "model.bin").exists()
            models.append(
                ModelInfo(
                    name=model_name,
                    type="stt",
                    size_mb=WHISPER_MODELS[model_name]["size_mb"],
                    installed=installed,
                    path=model_dir if installed else None,
                )
            )

        # Check Piper voices
        tts_path = self._get_tts_path()
        for voice_name in PIPER_VOICES:
            voice_file = tts_path / f"{voice_name}.onnx"
            installed = voice_file.exists()
            models.append(
                ModelInfo(
                    name=voice_name,
                    type="tts",
                    size_mb=PIPER_VOICES[voice_name]["size_mb"],
                    installed=installed,
                    path=voice_file if installed else None,
                )
            )

        return models

    def is_model_installed(self, name: str) -> bool:
        """Check if a model is installed.

        Args:
            name: Model or voice name.

        Returns:
            True if installed.
        """
        for model in self.list_installed_models():
            if model.name == name:
                return model.installed
        return False

    def get_model_path(self, name: str) -> Optional[Path]:
        """Get the path for a model.

        Args:
            name: Model or voice name.

        Returns:
            Path to model, or None if not installed.
        """
        for model in self.list_installed_models():
            if model.name == name:
                return model.path
        return None

    async def download_whisper_model(
        self,
        model_name: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        """Download a Whisper model.

        Args:
            model_name: Name of the model (e.g., "tiny.en").
            progress_callback: Optional callback for progress updates.

        Returns:
            Path to downloaded model directory.
        """
        if model_name not in WHISPER_MODELS:
            raise ValueError(f"Unknown Whisper model: {model_name}")

        model_info = WHISPER_MODELS[model_name]
        model_dir = self._get_stt_path() / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("downloading_whisper_model", model=model_name)

        async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
            for filename in model_info["files"]:
                url = model_info["url"] + filename
                file_path = model_dir / filename

                if file_path.exists():
                    self.logger.debug("file_exists_skipping", file=filename)
                    continue

                await self._download_file(
                    client, url, file_path, progress_callback
                )

        self.logger.info("whisper_model_downloaded", model=model_name, path=str(model_dir))
        return model_dir

    async def download_piper_voice(
        self,
        voice_name: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> Path:
        """Download a Piper voice.

        Args:
            voice_name: Name of the voice (e.g., "en_US-lessac-medium").
            progress_callback: Optional callback for progress updates.

        Returns:
            Path to downloaded voice file.
        """
        if voice_name not in PIPER_VOICES:
            raise ValueError(f"Unknown Piper voice: {voice_name}")

        voice_info = PIPER_VOICES[voice_name]
        voice_dir = self._get_tts_path()
        voice_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("downloading_piper_voice", voice=voice_name)

        async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
            for filename in voice_info["files"]:
                url = voice_info["url"] + filename
                file_path = voice_dir / filename

                if file_path.exists():
                    self.logger.debug("file_exists_skipping", file=filename)
                    continue

                await self._download_file(
                    client, url, file_path, progress_callback
                )

        voice_path = voice_dir / f"{voice_name}.onnx"
        self.logger.info("piper_voice_downloaded", voice=voice_name, path=str(voice_path))
        return voice_path

    async def _download_file(
        self,
        client: httpx.AsyncClient,
        url: str,
        path: Path,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """Download a single file.

        Args:
            client: HTTP client.
            url: URL to download from.
            path: Path to save to.
            progress_callback: Optional progress callback.
        """
        filename = path.name
        self.logger.debug("downloading_file", url=url, path=str(path))

        try:
            async with client.stream("GET", url) as response:
                response.raise_for_status()

                total = int(response.headers.get("content-length", 0))
                downloaded = 0

                # Write to temp file first
                temp_path = path.with_suffix(".tmp")

                with open(temp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total > 0:
                            progress_callback(
                                DownloadProgress(
                                    filename=filename,
                                    total_bytes=total,
                                    downloaded_bytes=downloaded,
                                    percent=(downloaded / total) * 100,
                                )
                            )

                # Move to final location
                shutil.move(str(temp_path), str(path))

        except Exception as e:
            self.logger.error("download_failed", url=url, error=str(e))
            # Clean up temp file if exists
            temp_path = path.with_suffix(".tmp")
            if temp_path.exists():
                temp_path.unlink()
            raise

    def delete_model(self, name: str) -> bool:
        """Delete an installed model.

        Args:
            name: Model or voice name.

        Returns:
            True if deleted, False if not found.
        """
        # Check if it's a Whisper model
        if name in WHISPER_MODELS:
            model_dir = self._get_stt_path() / name
            if model_dir.exists():
                shutil.rmtree(model_dir)
                self.logger.info("model_deleted", name=name)
                return True

        # Check if it's a Piper voice
        if name in PIPER_VOICES:
            tts_path = self._get_tts_path()
            deleted = False
            for suffix in [".onnx", ".onnx.json"]:
                file_path = tts_path / f"{name}{suffix}"
                if file_path.exists():
                    file_path.unlink()
                    deleted = True
            if deleted:
                self.logger.info("voice_deleted", name=name)
                return True

        return False

    def get_total_size(self) -> int:
        """Get total size of all installed models in bytes.

        Returns:
            Total size in bytes.
        """
        total = 0

        if self.base_path.exists():
            for path in self.base_path.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size

        return total
