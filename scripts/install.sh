#!/bin/bash
# HARO Installation Script for SteamOS/Linux
#
# This script installs HARO and its dependencies on Steam Deck or other Linux systems.
# Run with: curl -sSL https://raw.githubusercontent.com/user/haro/main/scripts/install.sh | bash
#
# Or clone and run locally:
#   git clone https://github.com/user/haro.git
#   cd haro
#   ./scripts/install.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Installation paths
INSTALL_DIR="${HARO_INSTALL_DIR:-$HOME/.local/share/haro}"
BIN_DIR="${HARO_BIN_DIR:-$HOME/.local/bin}"
CONFIG_DIR="${HARO_CONFIG_DIR:-$HOME/.config/haro}"
CONTEXT_DIR="${HARO_CONTEXT_DIR:-$HOME/.context}"
MODELS_DIR="${HARO_MODELS_DIR:-$HOME/.cache/haro/models}"
VENV_DIR="$INSTALL_DIR/venv"

# Python version requirement
PYTHON_MIN_VERSION="3.11"

# Log functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Steam Deck
is_steam_deck() {
    if [[ -f /etc/os-release ]]; then
        grep -q "steamos" /etc/os-release && return 0
    fi
    return 1
}

# Check Python version
check_python() {
    log_info "Checking Python installation..."

    # Try python3 first, then python
    for cmd in python3 python; do
        if command -v $cmd &> /dev/null; then
            version=$($cmd -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
            if [[ "$(printf '%s\n' "$PYTHON_MIN_VERSION" "$version" | sort -V | head -n1)" == "$PYTHON_MIN_VERSION" ]]; then
                PYTHON_CMD=$cmd
                log_success "Found Python $version ($cmd)"
                return 0
            fi
        fi
    done

    log_error "Python $PYTHON_MIN_VERSION or higher is required"
    log_info "On SteamOS, you may need to enable developer mode and install Python"
    return 1
}

# Check and install system dependencies
check_dependencies() {
    log_info "Checking system dependencies..."

    local missing=()

    # Check for required commands
    for cmd in git pip; do
        if ! command -v $cmd &> /dev/null; then
            missing+=($cmd)
        fi
    done

    # Check for audio libraries
    if ! pkg-config --exists portaudio-2.0 2>/dev/null; then
        missing+=("portaudio")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        log_warn "Missing dependencies: ${missing[*]}"

        if is_steam_deck; then
            log_info "On Steam Deck, some packages may need to be installed via flatpak or by enabling developer mode"
        else
            log_info "Install missing packages with your package manager:"
            log_info "  Arch: sudo pacman -S ${missing[*]}"
            log_info "  Ubuntu: sudo apt install ${missing[*]}"
            log_info "  Fedora: sudo dnf install ${missing[*]}"
        fi

        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        log_success "All system dependencies found"
    fi
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."

    mkdir -p "$INSTALL_DIR"
    mkdir -p "$BIN_DIR"
    mkdir -p "$CONFIG_DIR"
    mkdir -p "$CONTEXT_DIR"
    mkdir -p "$MODELS_DIR"

    log_success "Directories created"
}

# Create virtual environment
create_venv() {
    log_info "Creating Python virtual environment..."

    if [[ -d "$VENV_DIR" ]]; then
        log_warn "Virtual environment already exists, recreating..."
        rm -rf "$VENV_DIR"
    fi

    $PYTHON_CMD -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip wheel setuptools

    log_success "Virtual environment created at $VENV_DIR"
}

# Install HARO package
install_haro() {
    log_info "Installing HARO..."

    source "$VENV_DIR/bin/activate"

    # Check if we're in a git repo with HARO
    if [[ -f "pyproject.toml" ]] && grep -q "haro" pyproject.toml 2>/dev/null; then
        log_info "Installing from local source..."
        pip install -e ".[all]"
    else
        log_info "Installing from PyPI..."
        pip install haro[all]
    fi

    log_success "HARO installed"
}

# Create launcher script
create_launcher() {
    log_info "Creating launcher script..."

    cat > "$BIN_DIR/haro" << EOF
#!/bin/bash
# HARO launcher script
source "$VENV_DIR/bin/activate"
exec python -m haro "\$@"
EOF

    chmod +x "$BIN_DIR/haro"

    # Add to PATH if not already there
    if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
        log_info "Adding $BIN_DIR to PATH..."

        # Detect shell and update profile
        if [[ -f "$HOME/.bashrc" ]]; then
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.bashrc"
        fi
        if [[ -f "$HOME/.zshrc" ]]; then
            echo "export PATH=\"$BIN_DIR:\$PATH\"" >> "$HOME/.zshrc"
        fi

        export PATH="$BIN_DIR:$PATH"
    fi

    log_success "Launcher created at $BIN_DIR/haro"
}

# Create default configuration
create_config() {
    log_info "Creating default configuration..."

    if [[ ! -f "$CONFIG_DIR/config.yaml" ]]; then
        cat > "$CONFIG_DIR/config.yaml" << EOF
# HARO Configuration
# Generated by install.sh

audio:
  sample_rate: 16000
  channels: 1
  chunk_size: 1024
  input_device: null  # null for default
  output_device: null

wake:
  phrase: "HARO"
  sensitivity: 0.5
  cooldown: 2.0

vad:
  threshold: 0.02
  min_speech_duration: 0.5
  max_speech_duration: 30.0
  silence_duration: 1.5

stt:
  model: "tiny.en"
  device: "cpu"

tts:
  voice: "en_US-lessac-medium"
  speed: 1.0

api:
  model: "claude-sonnet-4-20250514"
  max_tokens: 1000
  temperature: 0.7
  timeout: 30

context:
  path: "$CONTEXT_DIR"
  history_turns: 10
  auto_save: true

logging:
  level: "INFO"
  file: "$CONFIG_DIR/haro.log"
EOF
        log_success "Configuration created at $CONFIG_DIR/config.yaml"
    else
        log_warn "Configuration already exists, skipping"
    fi
}

# Initialize context directory
init_context() {
    log_info "Initializing .context directory..."

    source "$VENV_DIR/bin/activate"
    python -m haro init-context --path "$CONTEXT_DIR" 2>/dev/null || true

    log_success "Context directory initialized at $CONTEXT_DIR"
}

# Download speech models
download_models() {
    log_info "Downloading speech models..."

    read -p "Download Whisper STT model (~75MB)? (Y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        source "$VENV_DIR/bin/activate"
        python -m haro download-model whisper tiny.en 2>/dev/null || log_warn "Model download failed, will download on first run"
    fi

    read -p "Download Piper TTS voice (~50MB)? (Y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        source "$VENV_DIR/bin/activate"
        python -m haro download-model piper en_US-lessac-medium 2>/dev/null || log_warn "Voice download failed, will download on first run"
    fi
}

# Install systemd service (optional)
install_service() {
    log_info "Setting up systemd service..."

    read -p "Install HARO as a user systemd service? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        return 0
    fi

    mkdir -p "$HOME/.config/systemd/user"

    cat > "$HOME/.config/systemd/user/haro.service" << EOF
[Unit]
Description=HARO Voice Assistant
After=sound.target

[Service]
Type=simple
ExecStart=$BIN_DIR/haro run --config $CONFIG_DIR/config.yaml
Restart=on-failure
RestartSec=5
Environment=ANTHROPIC_API_KEY=%h/.config/haro/api_key

[Install]
WantedBy=default.target
EOF

    systemctl --user daemon-reload

    log_success "Systemd service installed"
    log_info "To enable auto-start: systemctl --user enable haro"
    log_info "To start now: systemctl --user start haro"
}

# Setup API key
setup_api_key() {
    log_info "Setting up Anthropic API key..."

    if [[ -n "$ANTHROPIC_API_KEY" ]]; then
        log_success "API key found in environment"
        echo "$ANTHROPIC_API_KEY" > "$CONFIG_DIR/api_key"
        chmod 600 "$CONFIG_DIR/api_key"
    else
        log_warn "No API key found in ANTHROPIC_API_KEY environment variable"
        log_info "You can set it later by:"
        log_info "  1. export ANTHROPIC_API_KEY='your-key'"
        log_info "  2. Or save to $CONFIG_DIR/api_key"

        read -p "Enter your Anthropic API key (or press Enter to skip): " -r api_key
        if [[ -n "$api_key" ]]; then
            echo "$api_key" > "$CONFIG_DIR/api_key"
            chmod 600 "$CONFIG_DIR/api_key"
            log_success "API key saved"
        fi
    fi
}

# Test installation
test_installation() {
    log_info "Testing installation..."

    source "$VENV_DIR/bin/activate"

    if python -m haro status &>/dev/null; then
        log_success "HARO is working!"
    else
        log_warn "Basic test failed, but HARO may still work"
    fi
}

# Print completion message
print_completion() {
    echo
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}  HARO Installation Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo
    echo "Installation paths:"
    echo "  Install dir: $INSTALL_DIR"
    echo "  Config:      $CONFIG_DIR/config.yaml"
    echo "  Context:     $CONTEXT_DIR"
    echo "  Models:      $MODELS_DIR"
    echo
    echo "Commands:"
    echo "  haro status      - Check system status"
    echo "  haro run         - Start the assistant"
    echo "  haro --help      - Show all commands"
    echo
    if [[ -f "$HOME/.config/systemd/user/haro.service" ]]; then
        echo "Systemd service:"
        echo "  systemctl --user start haro   - Start HARO"
        echo "  systemctl --user enable haro  - Auto-start on login"
        echo
    fi
    echo "Documentation: https://github.com/user/haro#readme"
    echo
}

# Main installation flow
main() {
    echo
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}  HARO Installation Script${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo

    if is_steam_deck; then
        log_info "Steam Deck detected!"
    fi

    check_python
    check_dependencies
    create_directories
    create_venv
    install_haro
    create_launcher
    create_config
    init_context
    setup_api_key
    download_models
    install_service
    test_installation
    print_completion
}

# Run main function
main "$@"
