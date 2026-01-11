# CLAUDE.md - HARO Implementation Guide

## Project Overview

You are implementing **HARO** (Helpful Autonomous Responsive Operator), a lightweight voice AI assistant designed for dedicated hardware deployment on Steam Deck (SteamOS/Arch Linux).

HARO combines:
- **Local speech processing**: Whisper STT and Piper/SupertonicTTS running on CPU (<150MB each)
- **Cloud intelligence**: Claude API for reasoning and conversation
- **Persistent context**: The .context methodology for memory and knowledge management
- **Always-on listening**: Wake word detection ("HARO") for hands-free activation

## Documentation Structure

Before implementing anything, read these files in order:

1. `README.md` - Complete TINS specification with architecture, component designs, and code examples
2. `docs/Phase0.md` - Implementation roadmap with 8 phases, dependencies, and success criteria
3. `config/default.yaml` - Configuration schema covering all subsystems
4. `.context/substrate.md` - HARO's identity and capabilities
5. `.context/config/personality.md` - Response style guidelines
6. `.context/guidelines.md` - Operating constraints

## Implementation Approach

### Phase-Based Development

Follow the phases defined in `docs/Phase0.md` sequentially:

| Phase | Focus | Key Deliverables |
|-------|-------|------------------|
| 1 | Foundation & Audio | Project structure, audio I/O, VAD, config, events |
| 2 | Speech Processing | Whisper STT, Piper TTS, model management |
| 3 | Wake Word | Rolling buffer, "HARO" detection, activation feedback |
| 4 | State Machine | Agent loop, state transitions, interrupt handling |
| 5 | Claude API | Anthropic client, prompt construction, response parsing |
| 6 | .context Integration | Context assembly, session logging, knowledge management |
| 7 | Polish | Local commands, caching, startup/shutdown, optimization |
| 8 | Deployment | Steam Deck install script, systemd service, docs |

### For Each Phase

1. **Read the README.md section** for that component's design
2. **Create the implementation** in the appropriate `src/haro/` subdirectory
3. **Write tests** in `tests/` for the new functionality
4. **Update `__init__.py`** files to expose public interfaces
5. **Test the phase** before moving to the next

## Technical Constraints

### Target Environment
- Python 3.11+
- SteamOS 3.x (Arch Linux based)
- AMD Zen 2 CPU, 16GB RAM
- No GPU acceleration (CPU-only inference)

### Performance Targets
- Wake word detection: <200ms
- End-to-end response: <6 seconds
- CPU usage: <30% passive, <80% peak
- RAM: <500MB total

### Dependencies
Use the versions specified in `pyproject.toml`. Key packages:
- `faster-whisper` for STT (not openai-whisper)
- `piper-tts` for TTS
- `anthropic` for Claude API
- `pyaudio` or `sounddevice` for audio

## Code Style

- **Async/await** for all I/O operations
- **Type hints** on all functions
- **Dataclasses** for configuration and state
- **Structlog** for logging
- **YAML** for configuration files

### Example Pattern

```python
from dataclasses import dataclass
from typing import Optional
import asyncio
import structlog

logger = structlog.get_logger()

@dataclass
class ComponentConfig:
    """Configuration for this component."""
    setting: str
    value: int = 10

class Component:
    """Component description."""
    
    def __init__(self, config: ComponentConfig):
        self.config = config
        self.logger = logger.bind(component=self.__class__.__name__)
        
    async def initialize(self) -> None:
        """Initialize the component."""
        self.logger.info("initializing")
        # ...
        
    async def process(self, input: str) -> Optional[str]:
        """Process input and return result."""
        try:
            # Implementation
            return result
        except Exception as e:
            self.logger.error("processing_failed", error=str(e))
            raise
```

## Directory Structure

```
src/haro/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point (already created)
├── core/
│   ├── __init__.py
│   ├── agent.py         # Main agent state machine
│   ├── config.py        # Configuration loading
│   └── events.py        # Event system
├── audio/
│   ├── __init__.py
│   ├── capture.py       # Microphone input
│   ├── playback.py      # Speaker output
│   ├── vad.py           # Voice activity detection
│   └── wake.py          # Wake word detection
├── speech/
│   ├── __init__.py
│   ├── stt.py           # Whisper wrapper
│   └── tts.py           # Piper/Supertonic wrapper
├── intelligence/
│   ├── __init__.py
│   ├── client.py        # Claude API client
│   ├── prompts.py       # System prompt construction
│   └── parser.py        # Response parsing
├── context/
│   ├── __init__.py
│   ├── manager.py       # .context directory management
│   ├── session.py       # Session handling
│   └── knowledge.py     # Knowledge file operations
└── utils/
    ├── __init__.py
    ├── logging.py       # Logging setup
    └── audio_utils.py   # Audio format conversion
```

## Implementation Order for Phase 1

Start with these files in order:

1. `src/haro/utils/logging.py` - Structlog configuration
2. `src/haro/core/config.py` - YAML config loading with dataclasses
3. `src/haro/core/events.py` - Simple async event system
4. `src/haro/audio/capture.py` - Microphone streaming
5. `src/haro/audio/playback.py` - Speaker output
6. `src/haro/audio/vad.py` - Energy-based VAD

Then verify with:
```bash
python -m haro status  # Check dependencies
python -c "from haro.audio.capture import AudioCapture; print('OK')"
```

## Testing

Write tests alongside implementation:

```python
# tests/test_audio.py
import pytest
import numpy as np
from haro.audio.vad import VoiceActivityDetector, VADConfig

class TestVAD:
    def test_silence_detection(self):
        config = VADConfig(threshold=0.5)
        vad = VoiceActivityDetector(config)
        
        silence = np.zeros(1024, dtype=np.float32)
        result = vad.process(silence, timestamp=0.0)
        
        assert result.is_speech == False
```

Run with: `pytest tests/ -v`

## Key Design Decisions

1. **Wake word via Whisper**: We use Whisper on a rolling 2s buffer rather than a dedicated wake word model. This is simpler and more flexible, though slightly higher CPU.

2. **Piper over Supertonic**: Piper TTS is more readily available via pip. Supertonic can be added later.

3. **faster-whisper over openai-whisper**: CTranslate2 backend is significantly faster on CPU.

4. **Event-driven architecture**: Components communicate via events, not direct calls. This enables async operation and easier testing.

5. **.context for memory**: All persistent state goes in `.context/`. No database required.

## Commands Reference

```bash
# Development
pip install -e ".[dev]"      # Install in dev mode
python -m haro status        # Check system status
python -m haro init-context  # Create .context structure
python -m haro               # Run the agent

# Testing
pytest tests/ -v             # Run all tests
pytest tests/test_audio.py   # Run specific test file

# Linting
ruff check src/              # Lint code
mypy src/haro/               # Type checking
```

## When You're Stuck

1. Check the README.md for the component's intended design
2. Look at the code examples in README.md - they're meant to be close to final
3. Check Phase0.md for phase dependencies - you may need an earlier component
4. The config/default.yaml shows all expected configuration options

## Success Criteria

From Phase0.md, the MVP must:
- [ ] Detect wake word "HARO" within 200ms
- [ ] Transcribe speech with >90% accuracy
- [ ] Complete end-to-end response in <6 seconds
- [ ] Maintain multi-turn conversation context
- [ ] Log sessions to `.context/sessions/`
- [ ] Handle local commands without API calls
- [ ] Use <30% CPU during passive listening
- [ ] Use <500MB RAM total

---

**Start with Phase 1.** Read `README.md` first, then implement `src/haro/core/config.py`.
