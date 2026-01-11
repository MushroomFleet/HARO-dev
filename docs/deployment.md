# HARO Deployment Guide

This guide covers deploying HARO on Steam Deck and other Linux systems.

## Table of Contents

- [Quick Start](#quick-start)
- [Steam Deck Installation](#steam-deck-installation)
- [Manual Installation](#manual-installation)
- [Configuration](#configuration)
- [Running HARO](#running-haro)
- [Systemd Service](#systemd-service)
- [Troubleshooting](#troubleshooting)

## Quick Start

### One-Line Install

```bash
curl -sSL https://raw.githubusercontent.com/user/haro/main/scripts/install.sh | bash
```

### From Source

```bash
git clone https://github.com/user/haro.git
cd haro
./scripts/install.sh
```

## Steam Deck Installation

### Prerequisites

1. **Enable Desktop Mode**: Press the Steam button > Power > Switch to Desktop

2. **Enable Developer Mode** (optional, for system packages):
   - Settings > System > Enable Developer Mode
   - This allows installing system packages with `pacman`

3. **Set up Konsole**: Desktop Mode > Applications > System > Konsole

### Installation Steps

1. Open Konsole (terminal)

2. Install Python if not available:
   ```bash
   # Check Python version
   python3 --version

   # If missing or < 3.11, install via flatpak
   flatpak install flathub org.python.Python
   ```

3. Run the installer:
   ```bash
   curl -sSL https://raw.githubusercontent.com/user/haro/main/scripts/install.sh | bash
   ```

4. Set your API key:
   ```bash
   export ANTHROPIC_API_KEY='your-api-key-here'
   echo "export ANTHROPIC_API_KEY='your-api-key-here'" >> ~/.bashrc
   ```

5. Test the installation:
   ```bash
   haro status
   ```

### Steam Deck Audio Notes

- HARO automatically detects Steam Deck internal microphone and speakers
- External USB microphones/headsets are also supported
- For best results, use a headset to prevent echo

## Manual Installation

### System Requirements

- Python 3.11 or higher
- PortAudio library
- ~200MB disk space for models
- 512MB RAM minimum

### Dependencies

**Arch Linux / SteamOS:**
```bash
sudo pacman -S python python-pip portaudio
```

**Ubuntu / Debian:**
```bash
sudo apt install python3 python3-pip python3-venv portaudio19-dev
```

**Fedora:**
```bash
sudo dnf install python3 python3-pip portaudio-devel
```

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/user/haro.git
   cd haro
   ```

2. Create virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install HARO:
   ```bash
   pip install -e ".[all]"
   ```

4. Initialize context directory:
   ```bash
   python -m haro init-context
   ```

5. Download models:
   ```bash
   python -m haro download-model whisper tiny.en
   python -m haro download-model piper en_US-lessac-medium
   ```

## Configuration

### Configuration File

The default configuration is at `~/.config/haro/config.yaml`:

```yaml
# Audio settings
audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024
  input_device: null   # null = auto-detect
  output_device: null

# Wake word detection
wake:
  phrase: "HARO"
  sensitivity: 0.5     # 0.0 - 1.0, higher = more sensitive
  cooldown: 2.0        # seconds between activations

# Voice Activity Detection
vad:
  threshold: 0.02      # energy threshold
  min_speech_duration: 0.5
  max_speech_duration: 30.0
  silence_duration: 1.5

# Speech-to-Text
stt:
  model: "tiny.en"     # tiny, base, small, medium
  device: "cpu"

# Text-to-Speech
tts:
  voice: "en_US-lessac-medium"
  speed: 1.0

# Claude API
api:
  model: "claude-sonnet-4-20250514"
  max_tokens: 1000
  temperature: 0.7
  timeout: 30

# Context management
context:
  path: "~/.context"
  history_turns: 10
  auto_save: true

# Logging
logging:
  level: "INFO"        # DEBUG, INFO, WARNING, ERROR
  file: "~/.config/haro/haro.log"
```

### API Key Configuration

Set your Anthropic API key using one of these methods:

1. **Environment variable** (recommended):
   ```bash
   export ANTHROPIC_API_KEY='your-key-here'
   ```

2. **File**:
   ```bash
   echo 'your-key-here' > ~/.config/haro/api_key
   chmod 600 ~/.config/haro/api_key
   ```

3. **In config file** (not recommended):
   ```yaml
   api:
     key: "your-key-here"
   ```

### Audio Device Selection

List available devices:
```bash
haro status --devices
```

Set specific devices in config:
```yaml
audio:
  input_device: 0   # Device index from status output
  output_device: 2
```

Or by name:
```yaml
audio:
  input_device: "Built-in Microphone"
  output_device: "Built-in Speakers"
```

## Running HARO

### Interactive Mode

```bash
haro run
```

Press Ctrl+C to stop.

### With Custom Config

```bash
haro run --config /path/to/config.yaml
```

### Debug Mode

```bash
haro run --debug
```

### Check Status

```bash
haro status
```

Shows:
- Audio devices
- Model status
- API connection
- Context directory

## Systemd Service

### User Service (Recommended)

Install the service:
```bash
mkdir -p ~/.config/systemd/user
cp scripts/haro.service ~/.config/systemd/user/
systemctl --user daemon-reload
```

Control the service:
```bash
# Start HARO
systemctl --user start haro

# Stop HARO
systemctl --user stop haro

# Enable auto-start on login
systemctl --user enable haro

# Check status
systemctl --user status haro

# View logs
journalctl --user -u haro -f
```

### Environment Variables for Service

Create `~/.config/haro/env`:
```bash
ANTHROPIC_API_KEY=your-key-here
HARO_CONFIG=/home/user/.config/haro/config.yaml
```

## Troubleshooting

### Common Issues

#### "No audio devices found"

1. Check if audio devices are available:
   ```bash
   aplay -l   # List output devices
   arecord -l # List input devices
   ```

2. Install audio drivers if missing:
   ```bash
   sudo pacman -S pipewire pipewire-pulse
   ```

3. Restart audio service:
   ```bash
   systemctl --user restart pipewire
   ```

#### "Wake word not detected"

1. Check microphone is working:
   ```bash
   arecord -d 3 test.wav
   aplay test.wav
   ```

2. Adjust sensitivity in config:
   ```yaml
   wake:
     sensitivity: 0.7  # Increase for better detection
   ```

3. Ensure quiet environment for initial testing

#### "API connection failed"

1. Verify API key:
   ```bash
   echo $ANTHROPIC_API_KEY
   ```

2. Test API connection:
   ```bash
   curl -H "x-api-key: $ANTHROPIC_API_KEY" \
        https://api.anthropic.com/v1/messages
   ```

3. Check network connectivity

#### "Model download failed"

1. Check disk space:
   ```bash
   df -h ~/.cache/haro
   ```

2. Retry download:
   ```bash
   haro download-model whisper tiny.en --force
   ```

3. Manual download: Visit model repository and place files in `~/.cache/haro/models/`

#### High CPU Usage

1. Use smaller model:
   ```yaml
   stt:
     model: "tiny.en"  # Instead of "small" or "medium"
   ```

2. Increase VAD threshold:
   ```yaml
   vad:
     threshold: 0.05
   ```

3. Check for background processes:
   ```bash
   top -p $(pgrep -f haro)
   ```

### Steam Deck Specific

#### Audio in Gaming Mode

HARO works best in Desktop Mode. For Gaming Mode:

1. Add HARO as a non-Steam game
2. Set launch options: `konsole -e haro run`

#### Power Management

Battery optimization tips:
- Use `tiny.en` model
- Increase VAD threshold
- Enable sleep mode when idle

```yaml
vad:
  threshold: 0.03
context:
  session_timeout: 300  # Sleep after 5 minutes idle
```

### Getting Help

- Check logs: `~/.config/haro/haro.log`
- Debug mode: `haro run --debug`
- GitHub Issues: https://github.com/user/haro/issues
- Community: https://discord.gg/haro

## Uninstallation

Remove HARO completely:

```bash
# Stop service if running
systemctl --user stop haro
systemctl --user disable haro

# Remove files
rm -rf ~/.local/share/haro
rm -rf ~/.config/haro
rm -rf ~/.cache/haro
rm ~/.local/bin/haro
rm ~/.config/systemd/user/haro.service

# Optionally remove context
rm -rf ~/.context
```
