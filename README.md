# HARO - Helpful Autonomous Responsive Operator

<p align="center">
  <img src="https://img.shields.io/badge/Platform-Steam%20Deck-1a9fff?style=for-the-badge&logo=steam" alt="Steam Deck">
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?style=for-the-badge&logo=windows" alt="Windows">
  <img src="https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p>

<p align="center">
  <b>A lightweight, always-listening voice AI assistant designed for dedicated hardware deployment.</b>
</p>

---

## Overview

**HARO** is a voice-activated AI assistant that combines local speech processing with cloud intelligence. Built specifically for **Steam Deck** deployment, HARO provides hands-free interaction through wake word detection, speech recognition, and natural language responses.

### Key Features

- **Always-On Listening** - Wake word detection ("HARO") for hands-free activation
- **Local Speech Processing** - Whisper STT and Piper TTS running on CPU (<150MB each)
- **Cloud Intelligence** - Claude API integration for reasoning and conversation
- **Persistent Context** - Session memory and knowledge management via `.context` methodology
- **Immersive Feedback** - Context-aware acknowledgments and verbal cues during processing
- **Low Resource Usage** - Optimized for <30% CPU passive, <500MB RAM total

---

## Target Platform: Steam Deck

<p align="center">
  <img src="https://img.shields.io/badge/SteamOS-3.x-1a9fff?style=flat-square" alt="SteamOS 3.x">
  <img src="https://img.shields.io/badge/CPU-AMD%20Zen%202-ED1C24?style=flat-square" alt="AMD Zen 2">
  <img src="https://img.shields.io/badge/RAM-16GB-purple?style=flat-square" alt="16GB RAM">
</p>

HARO is designed from the ground up for the **Steam Deck** handheld gaming PC:

| Requirement | Specification |
|-------------|---------------|
| **OS** | SteamOS 3.x (Arch Linux) |
| **CPU** | AMD Zen 2 (CPU-only inference) |
| **RAM** | 16GB |
| **Storage** | ~500MB for models |
| **Network** | Required for Claude API |

### Why Steam Deck?

- **Dedicated Hardware** - Always-on voice assistant without tying up your main PC
- **Portable** - Take HARO anywhere with built-in speakers and microphone
- **Linux Native** - Runs natively on SteamOS without compatibility layers
- **Power Efficient** - Optimized for battery-powered operation

---

## Windows Demo

A Windows portable executable is available for testing and development:

### [Download Latest Release](https://github.com/MushroomFleet/HARO-dev/releases)

#### Quick Start (Windows)

1. Download and extract `HARO-Windows.zip` from [Releases](https://github.com/MushroomFleet/HARO-dev/releases)
2. Copy `.env.example` to `.env` and add your API key:
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   ```
3. Check system status:
   ```cmd
   haro.exe status
   ```
4. Download speech models (first time only):
   ```cmd
   haro.exe download-model whisper tiny
   haro.exe download-model piper en_US-lessac-medium
   ```
5. Run HARO:
   ```cmd
   haro.exe run
   ```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        HARO Agent                           │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐        │
│  │ PASSIVE │→ │ ACTIVE  │→ │PROCESS- │→ │SPEAKING │        │
│  │(listen) │  │(record) │  │  ING    │  │ (TTS)   │        │
│  └─────────┘  └─────────┘  └─────────┘  └─────────┘        │
│       ↑                                      │              │
│       └──────────────────────────────────────┘              │
├─────────────────────────────────────────────────────────────┤
│  Audio I/O    │  Wake Word   │  STT/TTS     │  Claude API  │
│  (sounddevice)│  (Whisper)   │  (Whisper/   │  (OpenRouter)│
│               │              │   Piper)     │              │
└─────────────────────────────────────────────────────────────┘
```

### Conversation Flow

```
User: "HARO, what's the weather like?"
       ↓
HARO: "Let me check the weather" (immediate acknowledgment)
       ↓
HARO: "Getting the info now" (while waiting for API)
       ↓
HARO: "It looks sunny with a high of 72 degrees, HARO." (response with sign-off)
```

---

## Available Commands

| Command | Description |
|---------|-------------|
| `haro run` | Start the voice assistant |
| `haro status` | Check system and dependency status |
| `haro test-audio` | Test audio input/output |
| `haro test-stt` | Test speech-to-text |
| `haro test-tts` | Test text-to-speech |
| `haro test-wake` | Test wake word detection |
| `haro models` | List installed speech models |
| `haro download-model` | Download a speech model |
| `haro init-context` | Initialize .context directory |

---

## Voice Commands

Once running, HARO responds to these built-in commands:

| Command | Action |
|---------|--------|
| "HARO, stop" | Cancel current speech |
| "HARO, repeat" | Repeat last response |
| "HARO, louder/quieter" | Adjust volume |
| "HARO, what time is it" | Get current time |
| "HARO, what's the date" | Get current date |
| "HARO, status" | System status |
| "HARO, help" | List available commands |
| "HARO, goodbye" | Graceful shutdown |

---

## Configuration

HARO uses YAML configuration files:

```yaml
# config/default.yaml (excerpt)
audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024

wake:
  phrase: "HARO"
  threshold: 0.6
  confirmation_sound: true

api:
  provider: "openrouter"
  model: "anthropic/claude-sonnet-4"
  timeout: 30
```

### Environment Variables

Create a `.env` file with your API key:

```bash
# OpenRouter (recommended)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Or direct Anthropic API
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

---

## Development

### Prerequisites

- Python 3.11+
- pip or uv package manager
- Audio input/output devices

### Installation

```bash
# Clone the repository
git clone https://github.com/MushroomFleet/HARO-dev.git
cd HARO-dev

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v
```

### Building Windows Executable

```bash
python scripts/build_windows.py --clean
```

Output: `dist/HARO/haro.exe`

---

## Roadmap

- [x] Phase 1: Foundation & Audio I/O
- [x] Phase 2: Speech Processing (Whisper/Piper)
- [x] Phase 3: Wake Word Detection
- [x] Phase 4: Agent State Machine
- [x] Phase 5: Claude API Integration
- [x] Phase 6: .context Memory System
- [x] Phase 7: Polish & Optimization
- [ ] Phase 8: Steam Deck Deployment (systemd service, install script)

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Wake word latency | <200ms | ~150ms |
| End-to-end response | <6s | ~4-5s |
| CPU (passive) | <30% | ~15-20% |
| CPU (active) | <80% | ~60-70% |
| RAM usage | <500MB | ~400MB |

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [faster-whisper](https://github.com/guillaumekln/faster-whisper) - CTranslate2 Whisper implementation
- [Piper TTS](https://github.com/rhasspy/piper) - Fast, local neural text-to-speech
- [Anthropic Claude](https://www.anthropic.com/) - AI reasoning and conversation
- [OpenRouter](https://openrouter.ai/) - Multi-model API gateway

---

## 📚 Citation

### Academic Citation

If you use this codebase in your research or project, please cite:

```bibtex
@software{haro_voice_assistant,
  title = {HARO: Helpful Autonomous Responsive Operator - A Lightweight Voice AI Assistant},
  author = {Drift Johnson},
  year = {2025},
  url = {https://github.com/MushroomFleet/HARO-dev},
  version = {1.0.0}
}
```

### Donate:

[![Ko-Fi](https://cdn.ko-fi.com/cdn/kofi3.png?v=3)](https://ko-fi.com/driftjohnson)
