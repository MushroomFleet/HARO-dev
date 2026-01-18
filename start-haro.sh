#!/bin/bash
# HARO Launcher for Steam Deck
# Run this script to start HARO voice assistant

cd /home/deck/HARO-dev
source venv/bin/activate

# Load environment variables
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Run HARO with Steam Deck config
exec haro --config ~/.config/haro/config.yaml "$@"
