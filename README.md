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

**HARO** is a voice-activated AI assistant that combines local speech processing with cloud intelligence. Built specifically for **Steam Deck** deployment (1280x720p optimized), HARO provides hands-free interaction through wake word detection, speech recognition, and natural language responses.

### Key Features

- **Always-On Listening** - Wake word detection ("HARO") for hands-free activation
- **Local Speech Processing** - Whisper STT and Piper TTS running on CPU (<150MB each)
- **Cloud Intelligence** - OpenRouter API integration with configurable models (default: Gemini Flash)
- **Streaming Responses** - Sentences are spoken as they arrive for faster response times
- **Rich Console UI** - Real-time status display optimized for 720p displays
- **Persistent Context** - Session memory and knowledge management via `.context` methodology
- **Low Resource Usage** - Optimized for <30% CPU passive, <500MB RAM total

---

## Quick Start

### 1. Get an API Key

Sign up at [OpenRouter](https://openrouter.ai/) to get an API key. OpenRouter provides access to multiple AI models including Gemini, Claude, and GPT.

### 2. Configure Environment

Create a `.env` file (or copy from `.env.example`):

```bash
# Required: Your OpenRouter API key
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional: Choose your model (default shown)
OPENROUTER_MODEL=google/gemini-3-flash-preview:online
```

**Available Models:**
| Model | Description |
|-------|-------------|
| `google/gemini-3-flash-preview:online` | Fast, web-enabled (default) |
| `anthropic/claude-sonnet-4` | High quality reasoning |
| `openai/gpt-4o` | OpenAI's latest |

The `:online` suffix enables web search for up-to-date information.

### 3. Download Speech Models (First Time)

```bash
haro download-model whisper tiny
haro download-model en_GB-southern_english_female-low
```

The default voice is `en_GB-southern_english_female-low` (British female, 18MB). Alternative voices are available - run `haro models` to see all options.

### 4. Run HARO

```bash
haro
```

That's it! HARO will launch with the UI enabled by default.

---

## Target Platform: Steam Deck

<p align="center">
  <img src="https://img.shields.io/badge/SteamOS-3.x-1a9fff?style=flat-square" alt="SteamOS 3.x">
  <img src="https://img.shields.io/badge/CPU-AMD%20Zen%202-ED1C24?style=flat-square" alt="AMD Zen 2">
  <img src="https://img.shields.io/badge/RAM-16GB-purple?style=flat-square" alt="16GB RAM">
  <img src="https://img.shields.io/badge/Display-1280x720-orange?style=flat-square" alt="1280x720">
</p>

HARO is designed from the ground up for the **Steam Deck** handheld gaming PC:

| Requirement | Specification |
|-------------|---------------|
| **OS** | SteamOS 3.x (Arch Linux) or Windows |
| **CPU** | AMD Zen 2 (CPU-only inference) |
| **RAM** | 16GB |
| **Storage** | ~500MB for models |
| **Display** | 1280x720 (UI optimized) |
| **Network** | Required for API calls |

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
2. Copy `.env.example` to `.env` and add your API key
3. Check system status: `haro.exe status`
4. Download models: `haro.exe download-model whisper tiny`
5. Run HARO: `haro.exe`

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
│  Audio I/O    │  Wake Word   │  STT/TTS     │  LLM API     │
│  (sounddevice)│  (Whisper)   │  (Whisper/   │  (OpenRouter)│
│               │              │   Piper)     │              │
└─────────────────────────────────────────────────────────────┘
```

### Conversation Flow

```
User: "HARO"
       ↓
HARO: "Hello, HARO?" (wake confirmation, then starts listening)
       ↓
User: "What's the weather like?"
       ↓
HARO: "HARO heard: What's the weather..." (acknowledgment)
       ↓
HARO: "It looks sunny with a high of 72 degrees. HARO HARO." (streamed response)
```

### Streaming Response

HARO uses streaming to reduce time-to-first-speech:
- LLM responses are streamed sentence-by-sentence
- Each sentence is sent to TTS as soon as it's complete
- Typical first-word latency: **0.7-1.5 seconds** (vs 3-7s without streaming)

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `haro` | Start the voice assistant (with UI) |
| `haro --no-ui` | Run without the rich console UI |
| `haro status` | Check system and dependency status |
| `haro test-audio` | Test audio input/output |
| `haro test-stt` | Test speech-to-text |
| `haro test-tts` | Test text-to-speech |
| `haro test-wake` | Test wake word detection |
| `haro models` | List installed speech models |
| `haro download-model <type> <name>` | Download a speech model |
| `haro init-context` | Initialize .context directory |
| `haro --help` | Show all available commands |

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
| "HARO, new conversation" | Clear conversation history |
| "HARO, goodbye" | Graceful shutdown |

### Special Keywords

| Keyword | Effect |
|---------|--------|
| "ULTRATALK" | Request verbose, detailed responses |
| "ask Claude" / "search online" | Force cloud LLM with web search |

---

## Configuration

### Environment Variables (.env)

```bash
# Required: API Key
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Optional: Model selection (default: google/gemini-3-flash-preview:online)
OPENROUTER_MODEL=google/gemini-3-flash-preview:online

# Optional: Custom API endpoint
# OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

# Alternative: Direct Anthropic API
# ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### YAML Configuration

HARO also supports YAML configuration files:

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
  model: "google/gemini-3-flash-preview:online"
  timeout: 30
  max_tokens: 1024
```

Configuration is loaded in order (later overrides earlier):
1. `config/default.yaml` (bundled)
2. `/etc/haro/config.yaml` (system)
3. `~/.config/haro/config.yaml` (user)
4. `./haro.yaml` (local)

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
- [x] Phase 7.5: Streaming LLM Responses
- [ ] Phase 8: Steam Deck Deployment (systemd service, install script)

---

## Performance Targets

| Metric | Target | Current |
|--------|--------|---------|
| Wake word latency | <200ms | ~150ms |
| Time to first word (streaming) | <2s | ~0.7-1.5s |
| End-to-end response | <6s | ~3-5s |
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
- [Google Gemini](https://deepmind.google/technologies/gemini/) - Default LLM provider

---

## Citation

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
