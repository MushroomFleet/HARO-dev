"""Tests for HARO deployment utilities.

Tests for:
- Audio device detection
- Platform detection
- Device manager functionality
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from haro.utils.devices import (
    AudioDevice,
    DeviceManager,
    get_device_summary,
)


# =============================================================================
# AudioDevice Tests
# =============================================================================


class TestAudioDevice:
    """Tests for AudioDevice dataclass."""

    def test_create_device(self):
        """Test creating an audio device."""
        device = AudioDevice(
            index=0,
            name="Test Microphone",
            channels=1,
            sample_rate=16000.0,
            is_input=True,
            is_output=False,
        )
        assert device.index == 0
        assert device.name == "Test Microphone"
        assert device.channels == 1
        assert device.is_input is True
        assert device.is_output is False

    def test_display_name_input(self):
        """Test display name for input device."""
        device = AudioDevice(
            index=0,
            name="Mic",
            channels=1,
            sample_rate=16000.0,
            is_input=True,
            is_output=False,
        )
        assert "input" in device.display_name
        assert "output" not in device.display_name

    def test_display_name_output(self):
        """Test display name for output device."""
        device = AudioDevice(
            index=0,
            name="Speakers",
            channels=2,
            sample_rate=44100.0,
            is_input=False,
            is_output=True,
        )
        assert "output" in device.display_name
        assert "input" not in device.display_name

    def test_display_name_both(self):
        """Test display name for input/output device."""
        device = AudioDevice(
            index=0,
            name="USB Headset",
            channels=2,
            sample_rate=44100.0,
            is_input=True,
            is_output=True,
        )
        assert "input" in device.display_name
        assert "output" in device.display_name

    def test_display_name_default(self):
        """Test display name with default marker."""
        device = AudioDevice(
            index=0,
            name="Device",
            channels=1,
            sample_rate=16000.0,
            is_input=True,
            is_output=False,
            is_default=True,
        )
        assert "[default]" in device.display_name

    def test_host_api(self):
        """Test host API field."""
        device = AudioDevice(
            index=0,
            name="Device",
            channels=1,
            sample_rate=16000.0,
            is_input=True,
            is_output=False,
            host_api="ALSA",
        )
        assert device.host_api == "ALSA"


# =============================================================================
# DeviceManager Tests (without sounddevice)
# =============================================================================


class TestDeviceManagerNoSounddevice:
    """Tests for DeviceManager when sounddevice is not available."""

    def test_not_available(self):
        """Test manager reports not available when sounddevice missing."""
        with patch.dict("sys.modules", {"sounddevice": None}):
            # Force reimport
            manager = DeviceManager()
            manager._available = False
            manager._sounddevice = None

            assert manager.available is False

    def test_get_input_devices_unavailable(self):
        """Test get_input_devices returns empty when unavailable."""
        manager = DeviceManager()
        manager._available = False

        devices = manager.get_input_devices()

        assert devices == []

    def test_get_output_devices_unavailable(self):
        """Test get_output_devices returns empty when unavailable."""
        manager = DeviceManager()
        manager._available = False

        devices = manager.get_output_devices()

        assert devices == []

    def test_get_default_input_unavailable(self):
        """Test get_default_input returns None when unavailable."""
        manager = DeviceManager()
        manager._available = False

        device = manager.get_default_input()

        assert device is None

    def test_get_default_output_unavailable(self):
        """Test get_default_output returns None when unavailable."""
        manager = DeviceManager()
        manager._available = False

        device = manager.get_default_output()

        assert device is None

    def test_auto_select_devices_unavailable(self):
        """Test auto_select_devices returns None when unavailable."""
        manager = DeviceManager()
        manager._available = False

        input_idx, output_idx = manager.auto_select_devices()

        assert input_idx is None
        assert output_idx is None

    def test_test_device_unavailable(self):
        """Test test_device returns False when unavailable."""
        manager = DeviceManager()
        manager._available = False

        result = manager.test_device(0, is_input=True)

        assert result is False


# =============================================================================
# DeviceManager Tests (with mocked sounddevice)
# =============================================================================


class TestDeviceManagerMocked:
    """Tests for DeviceManager with mocked sounddevice."""

    @pytest.fixture
    def mock_sd(self):
        """Create mocked sounddevice module."""
        mock = MagicMock()

        # Mock query_devices
        mock.query_devices.return_value = [
            {
                "name": "Built-in Microphone",
                "max_input_channels": 1,
                "max_output_channels": 0,
                "default_samplerate": 16000.0,
                "hostapi": 0,
            },
            {
                "name": "Built-in Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
                "hostapi": 0,
            },
            {
                "name": "USB Headset",
                "max_input_channels": 1,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
                "hostapi": 0,
            },
        ]

        # Mock default devices
        mock.default.device = [0, 1]  # [input, output]

        # Mock host APIs
        mock.query_hostapis.return_value = [
            {"name": "ALSA"},
            {"name": "PulseAudio"},
        ]

        return mock

    @pytest.fixture
    def manager(self, mock_sd):
        """Create DeviceManager with mocked sounddevice."""
        manager = DeviceManager()
        manager._sounddevice = mock_sd
        manager._available = True
        return manager

    def test_available(self, manager):
        """Test manager reports available."""
        assert manager.available is True

    def test_get_input_devices(self, manager):
        """Test getting input devices."""
        devices = manager.get_input_devices()

        assert len(devices) == 2
        names = [d.name for d in devices]
        assert "Built-in Microphone" in names
        assert "USB Headset" in names

    def test_get_output_devices(self, manager):
        """Test getting output devices."""
        devices = manager.get_output_devices()

        assert len(devices) == 2
        names = [d.name for d in devices]
        assert "Built-in Speakers" in names
        assert "USB Headset" in names

    def test_get_default_input(self, manager):
        """Test getting default input device."""
        device = manager.get_default_input()

        assert device is not None
        assert device.name == "Built-in Microphone"
        assert device.is_default is True

    def test_get_default_output(self, manager):
        """Test getting default output device."""
        device = manager.get_default_output()

        assert device is not None
        assert device.name == "Built-in Speakers"
        assert device.is_default is True

    def test_find_device_by_name(self, manager):
        """Test finding device by name."""
        device = manager.find_device_by_name("USB")

        assert device is not None
        assert "USB" in device.name

    def test_find_device_by_name_not_found(self, manager):
        """Test finding non-existent device."""
        device = manager.find_device_by_name("NonExistent")

        assert device is None

    def test_find_device_input_only(self, manager):
        """Test finding device with input_only filter."""
        device = manager.find_device_by_name("Speakers", input_only=True)

        assert device is None  # Speakers is output only

    def test_find_device_output_only(self, manager):
        """Test finding device with output_only filter."""
        device = manager.find_device_by_name("Microphone", output_only=True)

        assert device is None  # Microphone is input only

    def test_auto_select_devices(self, manager):
        """Test auto device selection."""
        input_idx, output_idx = manager.auto_select_devices()

        assert input_idx == 0
        assert output_idx == 1

    def test_get_platform_info(self, manager):
        """Test getting platform info."""
        info = manager.get_platform_info()

        assert "platform" in info
        assert "sounddevice_available" in info
        assert info["sounddevice_available"] is True
        assert "host_apis" in info


class TestDeviceManagerSteamDeck:
    """Tests for Steam Deck specific functionality."""

    @pytest.fixture
    def mock_sd_steam_deck(self):
        """Create mocked sounddevice for Steam Deck."""
        mock = MagicMock()

        mock.query_devices.return_value = [
            {
                "name": "acp_pdm Internal Microphone",
                "max_input_channels": 2,
                "max_output_channels": 0,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
            {
                "name": "acp_rn Internal Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
                "default_samplerate": 48000.0,
                "hostapi": 0,
            },
            {
                "name": "USB Audio Device",
                "max_input_channels": 1,
                "max_output_channels": 2,
                "default_samplerate": 44100.0,
                "hostapi": 0,
            },
        ]

        mock.default.device = [2, 2]  # USB as default
        mock.query_hostapis.return_value = [{"name": "ALSA"}]

        return mock

    def test_auto_select_prefers_internal(self, mock_sd_steam_deck):
        """Test that auto_select prefers internal devices on Steam Deck."""
        manager = DeviceManager()
        manager._sounddevice = mock_sd_steam_deck
        manager._available = True

        # Mock Steam Deck detection
        with patch.object(manager, "_is_steam_deck", return_value=True):
            input_idx, output_idx = manager.auto_select_devices()

        # Should prefer acp devices (index 0, 1) over USB (index 2)
        assert input_idx == 0
        assert output_idx == 1


# =============================================================================
# get_device_summary Tests
# =============================================================================


class TestGetDeviceSummary:
    """Tests for get_device_summary function."""

    def test_unavailable_message(self):
        """Test message when device detection unavailable."""
        with patch("haro.utils.devices.DeviceManager") as MockManager:
            mock_instance = MockManager.return_value
            mock_instance.available = False

            summary = get_device_summary()

            assert "not available" in summary.lower()

    def test_summary_content(self):
        """Test summary includes expected sections."""
        with patch("haro.utils.devices.DeviceManager") as MockManager:
            mock_instance = MockManager.return_value
            mock_instance.available = True
            mock_instance.get_platform_info.return_value = {
                "platform": "Linux",
                "platform_release": "5.15.0",
                "is_steam_deck": False,
            }
            mock_instance.get_input_devices.return_value = [
                AudioDevice(
                    index=0,
                    name="Test Mic",
                    channels=1,
                    sample_rate=16000.0,
                    is_input=True,
                    is_output=False,
                    is_default=True,
                ),
            ]
            mock_instance.get_output_devices.return_value = [
                AudioDevice(
                    index=1,
                    name="Test Speakers",
                    channels=2,
                    sample_rate=44100.0,
                    is_input=False,
                    is_output=True,
                    is_default=True,
                ),
            ]

            summary = get_device_summary()

            assert "Audio Devices" in summary
            assert "Input Devices" in summary
            assert "Output Devices" in summary
            assert "Test Mic" in summary
            assert "Test Speakers" in summary
            assert "Platform" in summary


# =============================================================================
# Integration Tests
# =============================================================================


class TestDeviceIntegration:
    """Integration tests for device utilities."""

    def test_device_manager_creation(self):
        """Test DeviceManager can be created."""
        manager = DeviceManager()
        # Should not raise, even if sounddevice not installed
        assert isinstance(manager, DeviceManager)

    def test_get_device_summary_no_crash(self):
        """Test get_device_summary doesn't crash."""
        # Should work regardless of sounddevice availability
        summary = get_device_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0


# =============================================================================
# Script and Service File Tests
# =============================================================================


class TestDeploymentFiles:
    """Tests for deployment script and service files."""

    def test_install_script_exists(self):
        """Test install script exists."""
        script_path = Path(__file__).parent.parent / "scripts" / "install.sh"
        assert script_path.exists(), f"Install script not found at {script_path}"

    def test_install_script_executable_content(self):
        """Test install script has proper shebang."""
        script_path = Path(__file__).parent.parent / "scripts" / "install.sh"
        with open(script_path) as f:
            first_line = f.readline()
        assert first_line.startswith("#!/bin/bash"), "Install script should have bash shebang"

    def test_service_file_exists(self):
        """Test systemd service file exists."""
        service_path = Path(__file__).parent.parent / "scripts" / "haro.service"
        assert service_path.exists(), f"Service file not found at {service_path}"

    def test_service_file_content(self):
        """Test systemd service file has required sections."""
        service_path = Path(__file__).parent.parent / "scripts" / "haro.service"
        with open(service_path) as f:
            content = f.read()

        assert "[Unit]" in content, "Service file should have [Unit] section"
        assert "[Service]" in content, "Service file should have [Service] section"
        assert "[Install]" in content, "Service file should have [Install] section"
        assert "ExecStart" in content, "Service file should have ExecStart"

    def test_deployment_docs_exist(self):
        """Test deployment documentation exists."""
        docs_path = Path(__file__).parent.parent / "docs" / "deployment.md"
        assert docs_path.exists(), f"Deployment docs not found at {docs_path}"

    def test_deployment_docs_content(self):
        """Test deployment docs have key sections."""
        docs_path = Path(__file__).parent.parent / "docs" / "deployment.md"
        with open(docs_path) as f:
            content = f.read()

        assert "Steam Deck" in content, "Docs should mention Steam Deck"
        assert "Installation" in content, "Docs should have installation section"
        assert "Configuration" in content, "Docs should have configuration section"
        assert "Troubleshooting" in content, "Docs should have troubleshooting section"
