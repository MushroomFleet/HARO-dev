# HARO - Device-Based Voice AI Agent

A lightweight, always-listening voice AI assistant designed for dedicated hardware deployment. HARO uses local speech processing with cloud-based intelligence, creating a responsive voice interface that feels like a dedicated assistant device.

Named after the iconic robot companion from Mobile Suit Gundam, HARO is designed to be helpful, responsive, and capable of maintaining contextual conversations over time.

## Project Overview

HARO transforms a Steam Deck (or similar Linux device) into a dedicated voice assistant by combining:

- **Local Speech Processing**: Whisper (STT) and SupertonicTTS (TTS) running on CPU under 150MB each
- **Cloud Intelligence**: Claude API for reasoning, conversation, and agentic capabilities
- **Persistent Context**: Integration with the .context methodology for memory and knowledge management
- **Always-On Listening**: Wake word detection ("HARO") for hands-free activation

### Target Hardware

- **Primary**: Steam Deck (SteamOS 3.x / Arch-based Linux)
- **CPU**: AMD Zen 2 (4 cores / 8 threads)
- **RAM**: 16GB DDR5
- **Storage**: Internal SSD or microSD
- **Audio**: Built-in microphone and speakers, or USB audio interface

### Design Philosophy

1. **Minimal Footprint**: Speech models run efficiently on CPU, leaving resources for other tasks
2. **Privacy-Conscious**: Audio is processed locally; only transcribed text goes to the API
3. **Context-Aware**: Maintains conversation history and learns from interactions via .context
4. **Offline Graceful**: Acknowledges when API is unavailable; core listening continues
5. **Single Purpose**: Designed as a dedicated assistant device, not a general-purpose app

## Functionality

### Core Voice Loop

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HARO Voice Loop                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────┐     ┌──────────────┐     ┌─────────────────────────┐ │
│   │   Mic   │────▶│ Wake Word    │────▶│ Voice Activity          │ │
│   │ Stream  │     │ Detection    │     │ Detection (VAD)         │ │
│   └─────────┘     │ ("HARO")     │     │ + Transcription         │ │
│                   └──────────────┘     └───────────┬─────────────┘ │
│                          │                         │               │
│                          │ (passive)               │ (active)      │
│                          ▼                         ▼               │
│                   ┌──────────────┐     ┌─────────────────────────┐ │
│                   │ Audio Buffer │     │ Whisper STT             │ │
│                   │ (2s rolling) │     │ (whisper.cpp / tiny)    │ │
│                   └──────────────┘     └───────────┬─────────────┘ │
│                                                    │               │
│                                                    ▼               │
│   ┌─────────────────────────────────────────────────────────────┐ │
│   │                    Agent Pipeline                            │ │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │ │
│   │  │ Context     │  │ Claude API  │  │ Response            │  │ │
│   │  │ Manager     │──│ Request     │──│ Parser              │  │ │
│   │  │ (.context)  │  │             │  │                     │  │ │
│   │  └─────────────┘  └─────────────┘  └─────────────────────┘  │ │
│   └───────────────────────────┬─────────────────────────────────┘ │
│                               │                                   │
│                               ▼                                   │
│   ┌─────────────────────────────────────────────────────────────┐ │
│   │                    Output Pipeline                           │ │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │ │
│   │  │ Response    │  │ Supertonic  │  │ Audio               │  │ │
│   │  │ Logger      │──│ TTS         │──│ Playback            │  │ │
│   │  │ (.context)  │  │             │  │                     │  │ │
│   │  └─────────────┘  └─────────────┘  └─────────────────────┘  │ │
│   └─────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Wake Word Detection

HARO uses a lightweight wake word detection system that runs continuously:

- **Wake Word**: "HARO" (phonetically distinct, low false-positive rate)
- **Detection Method**: Small Whisper model with keyword spotting on rolling buffer
- **Buffer Size**: 2 seconds of audio, refreshed continuously
- **Activation Feedback**: Audio chime or verbal acknowledgment ("Yes?", "I'm listening", etc.)
- **Timeout**: Returns to passive listening after 30 seconds of silence post-activation

### Conversation States

```
┌──────────────┐
│   PASSIVE    │◀─────────────────────────────────────┐
│  (listening  │                                      │
│  for wake)   │                                      │
└──────┬───────┘                                      │
       │ wake word detected                           │
       ▼                                              │
┌──────────────┐                                      │
│   ACTIVE     │──────────────────────────────────────┤
│  (recording  │ 30s silence timeout                  │
│   prompt)    │                                      │
└──────┬───────┘                                      │
       │ end of speech detected                       │
       ▼                                              │
┌──────────────┐                                      │
│  PROCESSING  │                                      │
│  (API call)  │                                      │
└──────┬───────┘                                      │
       │ response received                            │
       ▼                                              │
┌──────────────┐                                      │
│  SPEAKING    │──────────────────────────────────────┘
│  (TTS out)   │ speech complete → back to PASSIVE
└──────────────┘
       │ interrupt detected ("HARO")
       ▼
┌──────────────┐
│ INTERRUPTED  │───▶ return to ACTIVE
└──────────────┘
```

### Multi-Turn Conversation

HARO maintains conversation context across turns:

1. **Session Context**: Current conversation stored in memory, persisted to `.context/sessions/`
2. **History Window**: Last N turns (configurable, default 10) included in API context
3. **Context Files**: Relevant `.context/` documents loaded based on conversation topic
4. **Conversation Reset**: "HARO, new conversation" or 5 minutes of inactivity

### .context Integration

The `.context/` directory structure for HARO:

```
.context/
├── substrate.md              # Entry point, HARO personality and capabilities
├── sessions/                 # Conversation logs
│   ├── 2026-01-11_001.md    # Session transcript with metadata
│   ├── 2026-01-11_002.md
│   └── active.md            # Current active session (symlink or file)
├── knowledge/                # Persistent knowledge base
│   ├── user_preferences.md  # Learned user preferences
│   ├── facts.md             # Facts learned during conversations
│   └── corrections.md       # User corrections to HARO's understanding
├── config/                   # Runtime configuration
│   ├── personality.md       # HARO's personality traits and response style
│   ├── voice.md             # TTS voice settings
│   └── triggers.md          # Custom wake words and commands
└── guidelines.md             # Operating guidelines and constraints
```

### Agent Pipeline

When HARO receives a transcribed prompt, it:

1. **Context Assembly**:
   - Load current session context
   - Load relevant knowledge files (keyword matching)
   - Load personality and guidelines
   - Construct system prompt with .context content

2. **API Request**:
   - Send assembled context + user prompt to Claude API
   - Include conversation history (last N turns)
   - Request structured response (text + optional metadata)

3. **Response Processing**:
   - Parse response text for speech
   - Extract any knowledge updates (facts, preferences)
   - Identify if follow-up context loading is needed

4. **Output Generation**:
   - Log full response to session file
   - Update knowledge files if new information extracted
   - Generate speech via TTS
   - Play audio response

### Voice Commands

Built-in voice commands (processed locally before API):

| Command | Action |
|---------|--------|
| "HARO, stop" | Interrupt current speech |
| "HARO, new conversation" | Clear session, start fresh |
| "HARO, repeat that" | Replay last response |
| "HARO, louder/quieter" | Adjust TTS volume |
| "HARO, go to sleep" | Enter low-power passive mode |
| "HARO, what time is it" | Local time (no API call) |
| "HARO, goodbye" | End session, save context |

### Response Logging

Every interaction is logged to `.context/sessions/`:

```markdown
# Session 2026-01-11_001

## Metadata
- Started: 2026-01-11T14:32:00Z
- Device: steamdeck-haro
- Model: claude-sonnet-4-20250514

## Conversation

### Turn 1 [14:32:15]
**User**: What's the weather like in Bristol today?
**HARO**: I don't have real-time weather data, but I can help you check. Would you like me to remember Bristol as your location for future weather questions?

### Turn 2 [14:32:45]
**User**: Yes, please remember that.
**HARO**: Done! I've noted that you're in Bristol. In the future, when you ask about weather, I'll know you mean Bristol, UK.

## Knowledge Updates
- User location: Bristol, UK (added to user_preferences.md)

## Session Stats
- Turns: 2
- Duration: 45 seconds
- API calls: 2
```

## Technical Implementation

### Project Structure

```
haro/
├── README.md                 # This file (TINS specification)
├── pyproject.toml           # Python project configuration
├── requirements.txt         # Dependencies
├── .context/                # Context methodology integration
│   └── ...                  # As defined above
├── src/
│   └── haro/
│       ├── __init__.py
│       ├── __main__.py      # Entry point: python -m haro
│       ├── core/
│       │   ├── __init__.py
│       │   ├── agent.py     # Main agent loop and state machine
│       │   ├── config.py    # Configuration management
│       │   └── events.py    # Event system for component communication
│       ├── audio/
│       │   ├── __init__.py
│       │   ├── capture.py   # Microphone input handling
│       │   ├── playback.py  # Speaker output handling
│       │   ├── vad.py       # Voice activity detection
│       │   └── wake.py      # Wake word detection
│       ├── speech/
│       │   ├── __init__.py
│       │   ├── stt.py       # Whisper STT wrapper
│       │   └── tts.py       # SupertonicTTS wrapper
│       ├── intelligence/
│       │   ├── __init__.py
│       │   ├── client.py    # Claude API client
│       │   ├── prompts.py   # System prompt construction
│       │   └── parser.py    # Response parsing
│       ├── context/
│       │   ├── __init__.py
│       │   ├── manager.py   # .context file management
│       │   ├── session.py   # Session handling
│       │   └── knowledge.py # Knowledge base operations
│       └── utils/
│           ├── __init__.py
│           ├── logging.py   # Structured logging
│           └── audio_utils.py
├── tests/
│   ├── __init__.py
│   ├── test_audio.py
│   ├── test_speech.py
│   ├── test_agent.py
│   └── test_context.py
├── scripts/
│   ├── install.sh           # Installation script for SteamOS
│   ├── setup_audio.sh       # Audio device configuration
│   └── run.sh               # Launch script
└── config/
    ├── default.yaml         # Default configuration
    └── steamdeck.yaml       # Steam Deck specific config
```

### Dependencies

**Core Dependencies**:
```
# Audio
pyaudio>=0.2.13           # Audio I/O
numpy>=1.24.0             # Audio buffer handling
sounddevice>=0.4.6        # Alternative audio backend

# Speech
whisper.cpp bindings      # STT (via whispercpp or faster-whisper)
supertonic-tts            # TTS (or piper-tts as fallback)

# API
anthropic>=0.40.0         # Claude API client
httpx>=0.25.0             # HTTP client

# Context
pyyaml>=6.0               # YAML configuration
watchdog>=3.0.0           # File system monitoring

# Utilities
rich>=13.0.0              # Console output
structlog>=23.1.0         # Structured logging
```

**Model Files**:
- Whisper: `whisper-tiny.en` (~75MB) or `whisper-base.en` (~140MB)
- SupertonicTTS: English voice model (~100MB)

### Configuration

```yaml
# config/default.yaml

haro:
  # Device identification
  device_id: "haro-default"
  
  # Wake word settings
  wake:
    phrase: "haro"
    sensitivity: 0.5          # 0.0-1.0, lower = more sensitive
    confirmation_sound: true  # Play chime on activation
    confirmation_phrases:     # Random selection for variety
      - "Yes?"
      - "I'm here."
      - "Listening."
      - "How can I help?"
  
  # Audio settings
  audio:
    sample_rate: 16000
    channels: 1
    chunk_size: 1024
    input_device: null        # null = system default
    output_device: null
    
  # Voice activity detection
  vad:
    threshold: 0.5            # Energy threshold
    min_speech_duration: 0.5  # Minimum speech length (seconds)
    max_speech_duration: 30   # Maximum recording length
    silence_duration: 1.5     # Silence to end recording
    
  # Speech-to-text (Whisper)
  stt:
    model: "tiny.en"          # tiny.en, base.en, small.en
    model_path: "~/.cache/haro/models/"
    language: "en"
    compute_type: "int8"      # int8 for CPU efficiency
    
  # Text-to-speech
  tts:
    engine: "supertonic"      # supertonic, piper
    voice: "en_US-default"
    speed: 1.0
    volume: 0.8
    
  # API settings
  api:
    provider: "anthropic"
    model: "claude-sonnet-4-20250514"
    max_tokens: 1024
    temperature: 0.7
    timeout: 30
    
  # Context settings
  context:
    path: ".context/"
    history_turns: 10         # Conversation turns to include
    session_timeout: 300      # Seconds before auto-new-session
    auto_save: true
    
  # Logging
  logging:
    level: "INFO"
    file: "~/.local/share/haro/logs/haro.log"
    console: true
```

### Core Components

#### Agent State Machine (`core/agent.py`)

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional, Callable
import asyncio

class AgentState(Enum):
    PASSIVE = auto()      # Listening for wake word
    ACTIVE = auto()       # Recording user speech
    PROCESSING = auto()   # API call in progress
    SPEAKING = auto()     # TTS playback
    INTERRUPTED = auto()  # Wake word during speech
    SLEEPING = auto()     # Low-power mode
    ERROR = auto()        # Error state

@dataclass
class StateContext:
    """Context passed between state handlers"""
    transcript: Optional[str] = None
    response: Optional[str] = None
    error: Optional[Exception] = None
    session_id: Optional[str] = None

class HaroAgent:
    """Main agent state machine"""
    
    def __init__(self, config: Config):
        self.config = config
        self.state = AgentState.PASSIVE
        self.context = StateContext()
        
        # Component initialization
        self.audio_capture = AudioCapture(config.audio)
        self.audio_playback = AudioPlayback(config.audio)
        self.wake_detector = WakeWordDetector(config.wake)
        self.vad = VoiceActivityDetector(config.vad)
        self.stt = WhisperSTT(config.stt)
        self.tts = SupertonicTTS(config.tts)
        self.api_client = ClaudeClient(config.api)
        self.context_manager = ContextManager(config.context)
        
        # Event handlers
        self.on_state_change: Optional[Callable] = None
        self.on_transcript: Optional[Callable] = None
        self.on_response: Optional[Callable] = None
        
    async def run(self):
        """Main agent loop"""
        await self._initialize()
        
        while True:
            try:
                await self._process_state()
            except Exception as e:
                await self._handle_error(e)
                
    async def _process_state(self):
        """Process current state and transition"""
        handlers = {
            AgentState.PASSIVE: self._handle_passive,
            AgentState.ACTIVE: self._handle_active,
            AgentState.PROCESSING: self._handle_processing,
            AgentState.SPEAKING: self._handle_speaking,
            AgentState.INTERRUPTED: self._handle_interrupted,
            AgentState.SLEEPING: self._handle_sleeping,
        }
        
        handler = handlers.get(self.state)
        if handler:
            new_state = await handler()
            if new_state != self.state:
                await self._transition_to(new_state)
```

#### Wake Word Detection (`audio/wake.py`)

```python
import numpy as np
from dataclasses import dataclass
from typing import Optional
import asyncio

@dataclass
class WakeWordResult:
    detected: bool
    confidence: float
    timestamp: float

class WakeWordDetector:
    """Detect wake word using Whisper on rolling buffer"""
    
    def __init__(self, config: WakeConfig):
        self.config = config
        self.wake_phrase = config.phrase.lower()
        self.sensitivity = config.sensitivity
        self.buffer_duration = 2.0  # seconds
        self.buffer: Optional[np.ndarray] = None
        self.stt = None  # Lazy load whisper
        
    async def initialize(self, stt: WhisperSTT):
        """Initialize with shared STT instance"""
        self.stt = stt
        self.buffer = np.zeros(
            int(self.buffer_duration * 16000), 
            dtype=np.float32
        )
        
    def add_audio(self, chunk: np.ndarray):
        """Add audio chunk to rolling buffer"""
        chunk_len = len(chunk)
        self.buffer = np.roll(self.buffer, -chunk_len)
        self.buffer[-chunk_len:] = chunk
        
    async def detect(self) -> WakeWordResult:
        """Check buffer for wake word"""
        # Quick energy check first (avoid unnecessary transcription)
        energy = np.sqrt(np.mean(self.buffer ** 2))
        if energy < 0.01:  # Silence threshold
            return WakeWordResult(False, 0.0, 0.0)
            
        # Transcribe buffer
        result = await self.stt.transcribe(
            self.buffer, 
            prompt="HARO"  # Bias toward wake word
        )
        
        text = result.text.lower().strip()
        
        # Check for wake word
        if self.wake_phrase in text:
            # Calculate confidence based on match quality
            confidence = self._calculate_confidence(text, result)
            if confidence >= self.sensitivity:
                return WakeWordResult(True, confidence, result.timestamp)
                
        return WakeWordResult(False, 0.0, 0.0)
        
    def _calculate_confidence(self, text: str, result) -> float:
        """Calculate detection confidence"""
        # Factors: transcription confidence, word isolation, position
        base_conf = result.confidence if hasattr(result, 'confidence') else 0.8
        
        # Bonus if wake word is isolated or at start
        if text == self.wake_phrase:
            return min(1.0, base_conf + 0.2)
        if text.startswith(self.wake_phrase):
            return min(1.0, base_conf + 0.1)
            
        return base_conf
```

#### Context Manager (`context/manager.py`)

```python
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict
import yaml

class ContextManager:
    """Manage .context directory and files"""
    
    def __init__(self, config: ContextConfig):
        self.config = config
        self.base_path = Path(config.path).expanduser()
        self.current_session: Optional[Session] = None
        self._ensure_structure()
        
    def _ensure_structure(self):
        """Create .context directory structure if missing"""
        dirs = [
            self.base_path,
            self.base_path / "sessions",
            self.base_path / "knowledge",
            self.base_path / "config",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
            
        # Create default files if missing
        self._ensure_file("substrate.md", DEFAULT_SUBSTRATE)
        self._ensure_file("guidelines.md", DEFAULT_GUIDELINES)
        self._ensure_file("config/personality.md", DEFAULT_PERSONALITY)
        
    def assemble_context(self, user_input: str) -> str:
        """Assemble full context for API request"""
        parts = []
        
        # 1. Load substrate (entry point)
        parts.append(self._load_file("substrate.md"))
        
        # 2. Load personality
        parts.append(self._load_file("config/personality.md"))
        
        # 3. Load guidelines
        parts.append(self._load_file("guidelines.md"))
        
        # 4. Load relevant knowledge (keyword matching)
        relevant = self._find_relevant_knowledge(user_input)
        for file in relevant:
            parts.append(self._load_file(file))
            
        # 5. Load user preferences
        prefs = self._load_file("knowledge/user_preferences.md")
        if prefs:
            parts.append(prefs)
            
        # 6. Current session context (last N turns)
        if self.current_session:
            parts.append(self.current_session.get_history(
                self.config.history_turns
            ))
            
        return "\n\n---\n\n".join(filter(None, parts))
        
    def start_session(self) -> Session:
        """Start a new conversation session"""
        session_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        session_path = self.base_path / "sessions" / f"{session_id}.md"
        
        self.current_session = Session(
            id=session_id,
            path=session_path,
            started=datetime.now()
        )
        self.current_session.initialize()
        return self.current_session
        
    def log_turn(self, user_input: str, response: str):
        """Log a conversation turn"""
        if not self.current_session:
            self.start_session()
            
        self.current_session.add_turn(user_input, response)
        
        if self.config.auto_save:
            self.current_session.save()
            
    def update_knowledge(self, key: str, value: str, file: str = "facts.md"):
        """Update a knowledge file with new information"""
        knowledge_path = self.base_path / "knowledge" / file
        
        # Append to file with timestamp
        with open(knowledge_path, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"\n## {key} [{timestamp}]\n{value}\n")
```

#### Claude API Client (`intelligence/client.py`)

```python
from anthropic import Anthropic
from dataclasses import dataclass
from typing import Optional, List
import asyncio

@dataclass
class APIResponse:
    text: str
    model: str
    usage: dict
    stop_reason: str

class ClaudeClient:
    """Claude API client with retry and error handling"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.client = Anthropic()  # Uses ANTHROPIC_API_KEY env var
        self.model = config.model
        
    async def complete(
        self,
        system_context: str,
        messages: List[dict],
        user_input: str
    ) -> APIResponse:
        """Send completion request to Claude API"""
        
        # Build messages list
        all_messages = messages.copy()
        all_messages.append({
            "role": "user",
            "content": user_input
        })
        
        try:
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=self.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                system=system_context,
                messages=all_messages
            )
            
            return APIResponse(
                text=response.content[0].text,
                model=response.model,
                usage={
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                },
                stop_reason=response.stop_reason
            )
            
        except Exception as e:
            # Log error, return fallback response
            raise APIError(f"Claude API error: {e}")
```

### Audio Processing

#### Voice Activity Detection (`audio/vad.py`)

```python
import numpy as np
from dataclasses import dataclass
from enum import Enum

class VADState(Enum):
    SILENCE = "silence"
    SPEECH = "speech"
    TRAILING = "trailing"

@dataclass 
class VADResult:
    is_speech: bool
    energy: float
    duration: float

class VoiceActivityDetector:
    """Simple energy-based VAD with adaptive threshold"""
    
    def __init__(self, config: VADConfig):
        self.config = config
        self.threshold = config.threshold
        self.min_speech = config.min_speech_duration
        self.max_speech = config.max_speech_duration
        self.silence_duration = config.silence_duration
        
        # State tracking
        self.state = VADState.SILENCE
        self.speech_start: Optional[float] = None
        self.last_speech: Optional[float] = None
        self.noise_floor = 0.01
        
    def process(self, chunk: np.ndarray, timestamp: float) -> VADResult:
        """Process audio chunk and return VAD result"""
        energy = self._calculate_energy(chunk)
        
        # Adaptive threshold
        is_speech = energy > (self.noise_floor * self.threshold)
        
        # Update noise floor during silence
        if not is_speech:
            self.noise_floor = 0.95 * self.noise_floor + 0.05 * energy
            
        # State machine
        if self.state == VADState.SILENCE:
            if is_speech:
                self.state = VADState.SPEECH
                self.speech_start = timestamp
                self.last_speech = timestamp
                
        elif self.state == VADState.SPEECH:
            if is_speech:
                self.last_speech = timestamp
            else:
                self.state = VADState.TRAILING
                
        elif self.state == VADState.TRAILING:
            if is_speech:
                self.state = VADState.SPEECH
                self.last_speech = timestamp
            elif timestamp - self.last_speech > self.silence_duration:
                # Speech ended
                duration = self.last_speech - self.speech_start
                self.state = VADState.SILENCE
                return VADResult(
                    is_speech=False,
                    energy=energy,
                    duration=duration
                )
                
        # Check max duration
        if self.speech_start and timestamp - self.speech_start > self.max_speech:
            duration = timestamp - self.speech_start
            self.state = VADState.SILENCE
            return VADResult(is_speech=False, energy=energy, duration=duration)
            
        return VADResult(
            is_speech=self.state in (VADState.SPEECH, VADState.TRAILING),
            energy=energy,
            duration=timestamp - self.speech_start if self.speech_start else 0
        )
        
    def _calculate_energy(self, chunk: np.ndarray) -> float:
        """Calculate RMS energy of audio chunk"""
        return float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
```

### Installation and Setup

#### Steam Deck Installation (`scripts/install.sh`)

```bash
#!/bin/bash
# HARO Installation Script for Steam Deck (SteamOS)

set -e

echo "=== HARO Installation for Steam Deck ==="

# Check if running on Steam Deck
if [[ ! -f /etc/os-release ]] || ! grep -q "SteamOS" /etc/os-release; then
    echo "Warning: This script is optimized for SteamOS"
fi

# Disable read-only filesystem (Steam Deck specific)
echo "Disabling read-only filesystem..."
sudo steamos-readonly disable

# Install system dependencies
echo "Installing system dependencies..."
sudo pacman -Sy --noconfirm \
    python \
    python-pip \
    python-virtualenv \
    portaudio \
    alsa-utils \
    git

# Create HARO directory
HARO_DIR="$HOME/.local/share/haro"
mkdir -p "$HARO_DIR"
cd "$HARO_DIR"

# Create virtual environment
echo "Creating Python virtual environment..."
python -m venv venv
source venv/bin/activate

# Install HARO
echo "Installing HARO..."
pip install --upgrade pip
pip install haro  # or: pip install git+https://github.com/user/haro.git

# Download models
echo "Downloading speech models..."
haro download-models --stt whisper-tiny.en --tts supertonic-en

# Create .context directory
echo "Initializing .context..."
haro init-context

# Set up systemd service (optional)
echo "Setting up systemd service..."
mkdir -p "$HOME/.config/systemd/user"
cat > "$HOME/.config/systemd/user/haro.service" << EOF
[Unit]
Description=HARO Voice Assistant
After=sound.target

[Service]
Type=simple
ExecStart=$HARO_DIR/venv/bin/python -m haro
Restart=on-failure
Environment="ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}"

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable haro

# Re-enable read-only filesystem
echo "Re-enabling read-only filesystem..."
sudo steamos-readonly enable

echo "=== Installation Complete ==="
echo "Set your API key: export ANTHROPIC_API_KEY='your-key'"
echo "Start HARO: systemctl --user start haro"
echo "Or run directly: source $HARO_DIR/venv/bin/activate && python -m haro"
```

## Testing Scenarios

### Unit Tests

```python
# tests/test_wake.py
import pytest
import numpy as np
from haro.audio.wake import WakeWordDetector, WakeWordResult

class TestWakeWordDetector:
    
    @pytest.fixture
    def detector(self):
        config = WakeConfig(phrase="haro", sensitivity=0.5)
        return WakeWordDetector(config)
        
    def test_silence_no_detection(self, detector):
        """Silent audio should not trigger detection"""
        silence = np.zeros(32000, dtype=np.float32)
        detector.add_audio(silence)
        result = detector.detect()
        assert result.detected == False
        
    def test_wake_word_detected(self, detector, mock_stt):
        """Wake word should be detected with sufficient confidence"""
        # Mock STT to return "haro"
        mock_stt.transcribe.return_value = MockResult(text="haro", confidence=0.9)
        detector.stt = mock_stt
        
        # Add audio with speech-like energy
        audio = np.random.randn(32000).astype(np.float32) * 0.1
        detector.add_audio(audio)
        
        result = detector.detect()
        assert result.detected == True
        assert result.confidence >= 0.5
```

### Integration Tests

```python
# tests/test_agent.py
import pytest
from haro.core.agent import HaroAgent, AgentState

class TestAgentStateMachine:
    
    @pytest.fixture
    def agent(self, mock_config):
        return HaroAgent(mock_config)
        
    async def test_passive_to_active_on_wake(self, agent):
        """Agent should transition from PASSIVE to ACTIVE on wake word"""
        agent.state = AgentState.PASSIVE
        agent.wake_detector.detect = AsyncMock(
            return_value=WakeWordResult(True, 0.9, 0.0)
        )
        
        await agent._process_state()
        
        assert agent.state == AgentState.ACTIVE
        
    async def test_full_conversation_cycle(self, agent, mock_api):
        """Test complete conversation: wake → record → process → speak"""
        # Simulate wake word
        agent.state = AgentState.PASSIVE
        
        # ... run through full cycle
        # ... assert final state is PASSIVE
        # ... assert response was logged
```

### End-to-End Tests

```python
# tests/test_e2e.py

class TestEndToEnd:
    
    async def test_simple_conversation(self, haro_instance):
        """Test a simple question and answer"""
        # Inject audio of "HARO, what is two plus two?"
        # Wait for response
        # Verify TTS output contains "four"
        # Verify session logged correctly
        pass
        
    async def test_multi_turn_context(self, haro_instance):
        """Test that context is maintained across turns"""
        # Turn 1: "HARO, my name is Alex"
        # Turn 2: "HARO, what's my name?"
        # Verify response includes "Alex"
        pass
```

## Performance Goals

### Response Latency
- Wake word detection: < 200ms
- Speech-to-text (5 second utterance): < 2 seconds
- API round-trip: < 3 seconds (network dependent)
- Text-to-speech generation: < 1 second
- **Total end-to-end**: < 6 seconds for typical interaction

### Resource Usage
- CPU: < 30% average during passive listening
- CPU: < 80% peak during STT/TTS
- RAM: < 500MB total (including models)
- Storage: < 500MB for application + models

### Reliability
- Wake word false positive rate: < 1 per hour
- Wake word false negative rate: < 5%
- Session recovery after crash
- Graceful degradation when API unavailable

## Accessibility

### Voice Feedback
- Confirmation sounds for state changes
- Verbal acknowledgment options ("I'm listening", "Processing", etc.)
- Clear indication when HARO is speaking vs. listening

### Configuration
- Adjustable wake word sensitivity
- Configurable response verbosity
- Speed and volume controls for TTS

## Extended Features (Future)

These features are planned for future iterations:

1. **Ollama Integration**: Local LLM fallback when API is unavailable
2. **Custom Wake Words**: User-defined activation phrases
3. **Multi-Language Support**: Additional Whisper and TTS models
4. **Skill System**: Pluggable capabilities (weather, calendar, etc.)
5. **Web Dashboard**: Configuration and log viewing via browser
6. **Voice Identification**: Per-user context and preferences

## Development Guidelines

### Code Style
- Python 3.11+ with type hints
- Async/await for all I/O operations
- Structured logging with structlog
- Configuration via YAML files and environment variables

### Contribution Flow
1. Fork repository
2. Create feature branch
3. Implement with tests
4. Update .context documentation
5. Submit pull request

### Release Process
1. Version bump in pyproject.toml
2. Update CHANGELOG.md
3. Tag release
4. Build and publish to PyPI

---

**HARO** - Your dedicated voice AI companion. Built for focused interaction, designed for hardware deployment.
