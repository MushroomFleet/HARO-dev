# HARO Quick Start Guide for Steam Deck

## Starting HARO

### From Desktop Mode

Open a terminal (Konsole) and run:

```bash
cd ~/HARO-dev
./start-haro.sh
```

### With UI (Default)

The Rich UI displays:
- Current state (PASSIVE, ACTIVE, PROCESSING, SPEAKING)
- Audio levels and voice activity
- Conversation history
- System status

### Without UI

For headless/background operation:

```bash
./start-haro.sh --no-ui
```

---

## Using the Wake Word

HARO is always listening for its wake word: **"HARO"** (pronounced HAH-roh)

### Activation Flow

1. **Say "HARO"** - You'll hear a confirmation chime
2. **Speak your question** - HARO records until you pause
3. **Wait for response** - HARO processes and speaks the answer
4. **Interrupt anytime** - Say "HARO" again or "stop" to cancel

### Tips for Wake Word Detection

- Speak clearly and at normal volume
- A brief pause after "HARO" helps detection
- Works best within 1-2 meters of the Steam Deck
- Reduce background noise for better accuracy

---

## Voice Commands

These commands work without calling the cloud API:

| Command | Action |
|---------|--------|
| "stop" / "be quiet" / "shut up" | Cancel current speech |
| "repeat that" / "say that again" | Repeat last response |
| "new conversation" / "start over" | Clear conversation history |
| "go to sleep" / "sleep mode" | Enter low-power mode |
| "goodbye" / "bye" | Shut down HARO |
| "what time is it" | Get current time |
| "louder" / "volume up" | Increase volume |
| "quieter" / "volume down" | Decrease volume |

---

## UI Overview

```
┌─────────────────────────────────────────┐
│  HARO v0.1.0          State: PASSIVE    │
├─────────────────────────────────────────┤
│  Audio: ▁▂▃▂▁▁▂▃▄▃▂▁  VAD: ○           │
├─────────────────────────────────────────┤
│  Conversation:                          │
│  > User: What's the weather like?       │
│  < HARO: Based on current data...       │
├─────────────────────────────────────────┤
│  Say "HARO" to activate                 │
└─────────────────────────────────────────┘
```

- **State**: PASSIVE (listening), ACTIVE (recording), PROCESSING (thinking), SPEAKING (responding)
- **Audio meter**: Shows microphone input levels
- **VAD indicator**: Lights up when voice activity detected
- **Conversation**: Shows recent exchanges

Press `Ctrl+C` to exit.

---

## Adding HARO as a Non-Steam Game

This lets you launch HARO directly from Gaming Mode.

### Step 1: Create a Launch Script

The `start-haro.sh` script is already created. Verify it's executable:

```bash
chmod +x ~/HARO-dev/start-haro.sh
```

### Step 2: Add to Steam

1. Open **Steam** (in Desktop Mode)
2. Click **Games** → **Add a Non-Steam Game to My Library**
3. Click **Browse...**
4. Navigate to `/home/deck/HARO-dev/`
5. Change filter from "Applications" to **"All Files"**
6. Select `start-haro.sh`
7. Click **Open**, then **Add Selected Programs**

### Step 3: Configure Launch Options

1. Find **HARO** in your Steam Library
2. Right-click → **Properties**
3. Set these options:

**Launch Options:**
```
konsole -e %command%
```

This opens HARO in a terminal window with the full UI.

**Alternative - Fullscreen Terminal:**
```
konsole --fullscreen -e %command%
```

### Step 4: Set Artwork (Optional)

1. In Steam Library, right-click HARO
2. Click **Manage** → **Set Custom Artwork**
3. Add custom images for grid/hero/logo

### Step 5: Launch from Gaming Mode

1. Switch to Gaming Mode
2. Find HARO in your Library (under "Non-Steam" category)
3. Press **Play** to launch
4. Use the Steam Deck's built-in microphone and speakers
5. Press **Steam + X** to bring up keyboard if needed
6. Hold **Steam** button and press **B** to exit

---

## Gaming Mode Tips

- **Audio**: HARO uses the Steam Deck's internal mic and speakers by default
- **Performance**: HARO uses ~30% CPU while listening, ~70% when processing
- **Battery**: Expect 2-3 hours of continuous use
- **Quick Exit**: Say "goodbye" or press `Ctrl+C` in the terminal

---

## Troubleshooting

### No audio input detected

```bash
# Check audio devices
haro status

# Test microphone
haro test-audio
```

### Wake word not detected

- Speak louder/clearer
- Check microphone isn't muted (Steam Deck quick settings)
- Try adjusting sensitivity in `~/.config/haro/config.yaml`:
  ```yaml
  wake:
    sensitivity: 0.4  # Lower = more sensitive (0.0-1.0)
  ```

### No API response

```bash
# Test API connection
source ~/HARO-dev/venv/bin/activate
source ~/HARO-dev/.env
haro status
```

Check that `OPENROUTER_API_KEY` is set in `~/HARO-dev/.env`

### Models not loaded

```bash
# Re-download models
haro download-model base.en
haro download-model en_GB-southern_english_female-low
```

---

## Configuration

User config location: `~/.config/haro/config.yaml`

Key settings:
- `wake.sensitivity`: Wake word sensitivity (0.0-1.0)
- `tts.volume`: Speech volume (0.0-1.0)
- `tts.speed`: Speech rate multiplier
- `api.model`: LLM model to use

---

## File Locations

| File | Purpose |
|------|---------|
| `~/HARO-dev/` | Main installation |
| `~/HARO-dev/.env` | API keys (keep secret!) |
| `~/HARO-dev/start-haro.sh` | Launch script |
| `~/.config/haro/config.yaml` | User configuration |
| `~/.cache/haro/models/` | Downloaded speech models |
| `~/HARO-dev/.context/` | Conversation memory |
