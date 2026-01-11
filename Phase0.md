# HARO - Phase 0: Project Overview

## Project Summary

HARO is a lightweight, always-listening voice AI assistant designed for dedicated hardware deployment on Steam Deck (SteamOS/Arch Linux). The system combines local speech processing (Whisper STT, SupertonicTTS) with cloud intelligence (Claude API) and persistent context management (.context methodology).

The project delivers a hands-free voice interface that feels like a dedicated assistant device rather than a phone app, with continuous wake word detection, multi-turn conversations, and automatic context logging.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HARO System Architecture                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                        Audio Layer                               │   │
│  │  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐    │   │
│  │  │   Mic     │  │  Speaker  │  │   VAD     │  │   Wake    │    │   │
│  │  │  Capture  │  │  Playback │  │ Detector  │  │  Detector │    │   │
│  │  └─────┬─────┘  └─────▲─────┘  └─────┬─────┘  └─────┬─────┘    │   │
│  └────────┼──────────────┼──────────────┼──────────────┼──────────┘   │
│           │              │              │              │               │
│  ┌────────┼──────────────┼──────────────┼──────────────┼──────────┐   │
│  │        ▼              │              ▼              ▼          │   │
│  │                      Speech Layer                              │   │
│  │  ┌─────────────────────────┐  ┌─────────────────────────┐     │   │
│  │  │     Whisper STT         │  │    SupertonicTTS        │     │   │
│  │  │   (whisper.cpp/tiny)    │  │                         │     │   │
│  │  └───────────┬─────────────┘  └───────────▲─────────────┘     │   │
│  └──────────────┼────────────────────────────┼───────────────────┘   │
│                 │                            │                        │
│  ┌──────────────┼────────────────────────────┼───────────────────┐   │
│  │              ▼         Agent Core         │                   │   │
│  │  ┌─────────────────┐  ┌─────────────────┐ │ ┌───────────────┐ │   │
│  │  │  State Machine  │◀▶│  Event System   │─┼─│ Command       │ │   │
│  │  │                 │  │                 │ │ │ Parser        │ │   │
│  │  └────────┬────────┘  └─────────────────┘ │ └───────────────┘ │   │
│  └───────────┼───────────────────────────────┼───────────────────┘   │
│              │                               │                        │
│  ┌───────────┼───────────────────────────────┼───────────────────┐   │
│  │           ▼        Intelligence Layer     ▲                   │   │
│  │  ┌─────────────────┐  ┌─────────────────┐ │ ┌───────────────┐ │   │
│  │  │ Context Manager │─▶│  Claude Client  │─┘ │ Response      │ │   │
│  │  │   (.context)    │  │     (API)       │   │ Parser        │ │   │
│  │  └─────────────────┘  └─────────────────┘   └───────────────┘ │   │
│  └───────────────────────────────────────────────────────────────┘   │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────────┐   │
│  │                      Persistence Layer                         │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐    │   │
│  │  │  Sessions   │  │  Knowledge  │  │   Configuration     │    │   │
│  │  │  (.context/ │  │  (.context/ │  │   (YAML files)      │    │   │
│  │  │  sessions/) │  │  knowledge/)│  │                     │    │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘    │   │
│  └───────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

## Phase Breakdown

### Phase 1: Foundation & Audio Pipeline
**Goal**: Establish project structure, audio capture/playback, and basic VAD
**Duration**: 3-4 days
**Dependencies**: None
**Deliverables**:
- Project skeleton with pyproject.toml, directory structure
- Audio capture from microphone (PyAudio/sounddevice)
- Audio playback to speakers
- Voice Activity Detection (energy-based)
- Basic event system for component communication
- Configuration management (YAML)

### Phase 2: Speech Processing
**Goal**: Integrate Whisper STT and SupertonicTTS for voice I/O
**Duration**: 3-4 days
**Dependencies**: Phase 1 (audio pipeline)
**Deliverables**:
- Whisper STT integration (whisper.cpp or faster-whisper)
- Model download and management
- SupertonicTTS integration (or Piper fallback)
- Audio format conversion utilities
- Speech → Text → Speech round-trip test

### Phase 3: Wake Word Detection
**Goal**: Implement always-on "HARO" wake word detection
**Duration**: 2-3 days
**Dependencies**: Phase 2 (Whisper STT)
**Deliverables**:
- Rolling audio buffer for continuous monitoring
- Wake word detection using Whisper
- Activation confirmation (chime/verbal)
- Sensitivity configuration
- False positive mitigation

### Phase 4: Agent State Machine
**Goal**: Build the core agent loop with state transitions
**Duration**: 3-4 days
**Dependencies**: Phase 3 (wake detection)
**Deliverables**:
- State machine (PASSIVE → ACTIVE → PROCESSING → SPEAKING)
- State transition handlers
- Interrupt handling (wake word during speech)
- Timeout management
- Error recovery

### Phase 5: Claude API Integration
**Goal**: Connect to Claude API for intelligent responses
**Duration**: 2-3 days
**Dependencies**: Phase 4 (state machine)
**Deliverables**:
- Anthropic SDK integration
- System prompt construction
- Message history management
- Response parsing
- Error handling and retry logic
- Rate limiting awareness

### Phase 6: .context Integration
**Goal**: Implement the .context methodology for persistent memory
**Duration**: 3-4 days
**Dependencies**: Phase 5 (API integration)
**Deliverables**:
- .context directory structure creation
- Context assembly for API requests
- Session logging (conversation transcripts)
- Knowledge file management
- Relevant context retrieval (keyword matching)
- Substrate and personality files

### Phase 7: Voice Commands & Polish
**Goal**: Add local voice commands, polish UX, optimize performance
**Duration**: 2-3 days
**Dependencies**: Phase 6 (.context integration)
**Deliverables**:
- Local command parsing ("HARO, stop", "HARO, repeat", etc.)
- Response caching for repeated questions
- Startup/shutdown sequences
- Performance profiling and optimization
- Comprehensive logging

### Phase 8: Steam Deck Deployment
**Goal**: Package and deploy to Steam Deck hardware
**Duration**: 2-3 days
**Dependencies**: Phase 7 (polish)
**Deliverables**:
- Installation script for SteamOS
- Systemd service configuration
- Audio device auto-detection
- Desktop mode integration
- Documentation and troubleshooting guide

## Phase Dependency Graph

```
Phase 1 (Foundation)
    │
    ▼
Phase 2 (Speech Processing)
    │
    ▼
Phase 3 (Wake Word)
    │
    ▼
Phase 4 (State Machine)
    │
    ▼
Phase 5 (Claude API)
    │
    ▼
Phase 6 (.context)
    │
    ▼
Phase 7 (Polish)
    │
    ▼
Phase 8 (Deployment)
```

## Success Criteria

### Functional Requirements
- [ ] Wake word "HARO" detected within 200ms
- [ ] Speech transcription accuracy > 90% for clear speech
- [ ] End-to-end response latency < 6 seconds
- [ ] Multi-turn conversations maintain context correctly
- [ ] Sessions logged to .context/sessions/ automatically
- [ ] Local commands work without API call

### Performance Requirements
- [ ] CPU usage < 30% during passive listening
- [ ] RAM usage < 500MB total
- [ ] False wake word activations < 1 per hour
- [ ] Recovery from errors without manual restart

### User Experience
- [ ] Clear audio feedback for state changes
- [ ] Natural conversation flow
- [ ] Graceful handling of API unavailability
- [ ] Easy configuration via YAML

## Technology Stack

### Runtime
- **Python**: 3.11+ (async/await, type hints)
- **OS**: SteamOS 3.x (Arch Linux based)

### Audio
- **PyAudio** or **sounddevice**: Audio I/O
- **NumPy**: Audio buffer manipulation

### Speech
- **whisper.cpp** (via faster-whisper or whispercpp): STT
- **SupertonicTTS** or **Piper**: TTS
- Model: whisper-tiny.en (~75MB)

### Intelligence
- **anthropic**: Claude API client
- **httpx**: HTTP client (async)

### Configuration & Persistence
- **PyYAML**: Configuration files
- **pathlib**: File system operations
- **Markdown**: .context files

### Development
- **pytest**: Testing
- **pytest-asyncio**: Async test support
- **structlog**: Structured logging
- **rich**: Console output

## Directory Structure

```
haro/
├── README.md                 # TINS specification
├── pyproject.toml           # Project configuration
├── requirements.txt         # Dependencies
├── docs/
│   ├── Phase0.md            # This file
│   ├── Phase1.md            # Foundation details
│   ├── Phase2.md            # Speech processing details
│   └── instruct/            # Active phase substages
├── .context/                # Context methodology
│   ├── substrate.md
│   ├── sessions/
│   ├── knowledge/
│   └── config/
├── src/haro/
│   ├── __init__.py
│   ├── __main__.py
│   ├── core/
│   ├── audio/
│   ├── speech/
│   ├── intelligence/
│   ├── context/
│   └── utils/
├── tests/
├── scripts/
└── config/
```

## Risk Assessment

### Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Whisper CPU performance | Medium | High | Use tiny.en model, optimize parameters |
| Wake word false positives | Medium | Medium | Tune sensitivity, add confirmation |
| Audio latency on Steam Deck | Low | High | Test early, use ALSA directly if needed |
| API rate limits | Low | Medium | Implement caching, exponential backoff |

### External Dependencies

| Dependency | Risk | Fallback |
|------------|------|----------|
| Claude API | Service unavailable | Cache responses, offline acknowledgment |
| Whisper models | Download failure | Bundle with installation |
| TTS engine | Compatibility issues | Piper as fallback |

## Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Foundation | 3-4 days | Week 1 |
| Phase 2: Speech | 3-4 days | Week 1-2 |
| Phase 3: Wake Word | 2-3 days | Week 2 |
| Phase 4: State Machine | 3-4 days | Week 2-3 |
| Phase 5: Claude API | 2-3 days | Week 3 |
| Phase 6: .context | 3-4 days | Week 3-4 |
| Phase 7: Polish | 2-3 days | Week 4 |
| Phase 8: Deployment | 2-3 days | Week 4-5 |

**Total Estimated Duration**: 4-5 weeks for MVP

## Next Steps

1. **Begin Phase 1**: Set up project structure and audio pipeline
2. **Validate Hardware**: Test audio I/O on Steam Deck early
3. **API Key Setup**: Ensure Anthropic API access configured
4. **Model Selection**: Finalize Whisper and TTS model choices

---

*This Phase 0 document provides the roadmap for HARO development. Each subsequent phase document (Phase1.md, Phase2.md, etc.) contains detailed implementation instructions, code examples, and verification steps.*
