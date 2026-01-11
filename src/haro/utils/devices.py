"""Audio device detection and management for HARO.

Provides cross-platform audio device enumeration and selection.
"""

from dataclasses import dataclass
from typing import Optional
import platform

from haro.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AudioDevice:
    """Represents an audio device."""

    index: int
    name: str
    channels: int
    sample_rate: float
    is_input: bool
    is_output: bool
    is_default: bool = False
    host_api: str = ""

    @property
    def display_name(self) -> str:
        """Get display name with type indicator."""
        types = []
        if self.is_input:
            types.append("input")
        if self.is_output:
            types.append("output")
        type_str = "/".join(types)
        default_str = " [default]" if self.is_default else ""
        return f"{self.name} ({type_str}){default_str}"


class DeviceManager:
    """Manage audio device detection and selection.

    Provides cross-platform device enumeration for:
    - Linux (ALSA, PulseAudio, PipeWire)
    - Windows (WASAPI, DirectSound)
    - macOS (CoreAudio)
    """

    def __init__(self) -> None:
        """Initialize device manager."""
        self._sounddevice = None
        self._available = False
        self.logger = logger.bind(component="DeviceManager")

        try:
            import sounddevice as sd
            self._sounddevice = sd
            self._available = True
        except ImportError:
            self.logger.warning("sounddevice not available, device detection disabled")

    @property
    def available(self) -> bool:
        """Check if device detection is available."""
        return self._available

    def get_input_devices(self) -> list[AudioDevice]:
        """Get all available input (microphone) devices.

        Returns:
            List of input AudioDevice objects.
        """
        if not self._available:
            return []

        devices = []
        try:
            all_devices = self._sounddevice.query_devices()
            default_input = self._sounddevice.default.device[0]

            for i, dev in enumerate(all_devices):
                if dev["max_input_channels"] > 0:
                    devices.append(AudioDevice(
                        index=i,
                        name=dev["name"],
                        channels=dev["max_input_channels"],
                        sample_rate=dev["default_samplerate"],
                        is_input=True,
                        is_output=dev["max_output_channels"] > 0,
                        is_default=(i == default_input),
                        host_api=self._get_host_api_name(dev.get("hostapi", 0)),
                    ))

        except Exception as e:
            self.logger.error("input_device_enumeration_failed", error=str(e))

        return devices

    def get_output_devices(self) -> list[AudioDevice]:
        """Get all available output (speaker) devices.

        Returns:
            List of output AudioDevice objects.
        """
        if not self._available:
            return []

        devices = []
        try:
            all_devices = self._sounddevice.query_devices()
            default_output = self._sounddevice.default.device[1]

            for i, dev in enumerate(all_devices):
                if dev["max_output_channels"] > 0:
                    devices.append(AudioDevice(
                        index=i,
                        name=dev["name"],
                        channels=dev["max_output_channels"],
                        sample_rate=dev["default_samplerate"],
                        is_input=dev["max_input_channels"] > 0,
                        is_output=True,
                        is_default=(i == default_output),
                        host_api=self._get_host_api_name(dev.get("hostapi", 0)),
                    ))

        except Exception as e:
            self.logger.error("output_device_enumeration_failed", error=str(e))

        return devices

    def get_default_input(self) -> Optional[AudioDevice]:
        """Get the default input device.

        Returns:
            Default input device or None.
        """
        for dev in self.get_input_devices():
            if dev.is_default:
                return dev
        return None

    def get_default_output(self) -> Optional[AudioDevice]:
        """Get the default output device.

        Returns:
            Default output device or None.
        """
        for dev in self.get_output_devices():
            if dev.is_default:
                return dev
        return None

    def find_device_by_name(
        self,
        name: str,
        input_only: bool = False,
        output_only: bool = False,
    ) -> Optional[AudioDevice]:
        """Find a device by name substring.

        Args:
            name: Substring to search for in device name.
            input_only: Only search input devices.
            output_only: Only search output devices.

        Returns:
            Matching device or None.
        """
        name_lower = name.lower()

        if input_only:
            devices = self.get_input_devices()
        elif output_only:
            devices = self.get_output_devices()
        else:
            devices = self.get_input_devices() + self.get_output_devices()

        for dev in devices:
            if name_lower in dev.name.lower():
                return dev

        return None

    def auto_select_devices(self) -> tuple[Optional[int], Optional[int]]:
        """Automatically select the best input and output devices.

        For Steam Deck, prefers internal microphone and speakers.

        Returns:
            Tuple of (input_device_index, output_device_index).
        """
        input_idx = None
        output_idx = None

        # Get defaults first
        default_input = self.get_default_input()
        default_output = self.get_default_output()

        if default_input:
            input_idx = default_input.index
        if default_output:
            output_idx = default_output.index

        # On Steam Deck, look for specific device names
        if self._is_steam_deck():
            # Prefer internal devices
            for dev in self.get_input_devices():
                if "acp" in dev.name.lower() or "internal" in dev.name.lower():
                    input_idx = dev.index
                    break

            for dev in self.get_output_devices():
                if "acp" in dev.name.lower() or "internal" in dev.name.lower():
                    output_idx = dev.index
                    break

        self.logger.info(
            "auto_selected_devices",
            input=input_idx,
            output=output_idx,
        )

        return input_idx, output_idx

    def test_device(
        self,
        device_index: int,
        is_input: bool = True,
        duration: float = 0.5,
    ) -> bool:
        """Test if a device is working.

        Args:
            device_index: Device index to test.
            is_input: True for input device, False for output.
            duration: Test duration in seconds.

        Returns:
            True if device works.
        """
        if not self._available:
            return False

        try:
            import numpy as np

            if is_input:
                # Record a short sample
                data = self._sounddevice.rec(
                    int(16000 * duration),
                    samplerate=16000,
                    channels=1,
                    device=device_index,
                    dtype=np.float32,
                )
                self._sounddevice.wait()
                return data is not None and len(data) > 0

            else:
                # Play a short silent sample
                data = np.zeros(int(16000 * duration), dtype=np.float32)
                self._sounddevice.play(
                    data,
                    samplerate=16000,
                    device=device_index,
                )
                self._sounddevice.wait()
                return True

        except Exception as e:
            self.logger.warning(
                "device_test_failed",
                device=device_index,
                is_input=is_input,
                error=str(e),
            )
            return False

    def get_platform_info(self) -> dict:
        """Get platform and audio system information.

        Returns:
            Dictionary with platform info.
        """
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "machine": platform.machine(),
            "is_steam_deck": self._is_steam_deck(),
            "sounddevice_available": self._available,
        }

        if self._available:
            try:
                info["host_apis"] = [
                    api["name"]
                    for api in self._sounddevice.query_hostapis()
                ]
            except Exception:
                info["host_apis"] = []

        return info

    def _get_host_api_name(self, api_index: int) -> str:
        """Get host API name by index.

        Args:
            api_index: Host API index.

        Returns:
            Host API name.
        """
        if not self._available:
            return ""

        try:
            apis = self._sounddevice.query_hostapis()
            if 0 <= api_index < len(apis):
                return apis[api_index]["name"]
        except Exception:
            pass

        return ""

    def _is_steam_deck(self) -> bool:
        """Check if running on Steam Deck.

        Returns:
            True if on Steam Deck.
        """
        # Check for SteamOS
        try:
            with open("/etc/os-release") as f:
                content = f.read().lower()
                if "steamos" in content:
                    return True
        except Exception:
            pass

        # Check for Steam Deck hardware
        try:
            with open("/sys/devices/virtual/dmi/id/product_name") as f:
                if "jupiter" in f.read().lower():
                    return True
        except Exception:
            pass

        return False


def get_device_summary() -> str:
    """Get a formatted summary of available audio devices.

    Returns:
        Human-readable device summary.
    """
    manager = DeviceManager()

    if not manager.available:
        return "Audio device detection not available (sounddevice not installed)"

    lines = ["Audio Devices:", ""]

    # Platform info
    info = manager.get_platform_info()
    lines.append(f"Platform: {info['platform']} {info['platform_release']}")
    if info.get("is_steam_deck"):
        lines.append("Running on: Steam Deck")
    lines.append("")

    # Input devices
    lines.append("Input Devices (Microphones):")
    inputs = manager.get_input_devices()
    if inputs:
        for dev in inputs:
            marker = " *" if dev.is_default else "  "
            lines.append(f"{marker}[{dev.index}] {dev.name}")
            lines.append(f"       Channels: {dev.channels}, Rate: {int(dev.sample_rate)} Hz")
    else:
        lines.append("  No input devices found")
    lines.append("")

    # Output devices
    lines.append("Output Devices (Speakers):")
    outputs = manager.get_output_devices()
    if outputs:
        for dev in outputs:
            marker = " *" if dev.is_default else "  "
            lines.append(f"{marker}[{dev.index}] {dev.name}")
            lines.append(f"       Channels: {dev.channels}, Rate: {int(dev.sample_rate)} Hz")
    else:
        lines.append("  No output devices found")
    lines.append("")

    lines.append("* = default device")

    return "\n".join(lines)
