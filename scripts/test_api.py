#!/usr/bin/env python3
"""Test script for HARO API integration.

Tests the Claude API connection using either direct Anthropic API or OpenRouter.

Usage:
    # With OpenRouter key as argument:
    python scripts/test_api.py --key sk-or-v1-...

    # With OpenRouter (set OPENROUTER_API_KEY env var first):
    set OPENROUTER_API_KEY=sk-or-v1-...
    python scripts/test_api.py

    # With direct Anthropic API:
    set ANTHROPIC_API_KEY=sk-ant-...
    python scripts/test_api.py
"""

import argparse
import asyncio
import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from haro.intelligence.client import ClaudeClient
from haro.core.config import APIConfig


async def test_api(api_key: str = None):
    """Test the API connection with a simple request."""
    print("=" * 60)
    print("HARO API Integration Test")
    print("=" * 60)

    # Check for API keys - command line takes precedence
    openrouter_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    # If key provided via command line, set it in environment for the client
    if api_key:
        if api_key.startswith("sk-or"):
            os.environ["OPENROUTER_API_KEY"] = api_key
            openrouter_key = api_key
        else:
            os.environ["ANTHROPIC_API_KEY"] = api_key
            anthropic_key = api_key

    if openrouter_key:
        print(f"\nUsing OpenRouter API (key: {openrouter_key[:20]}...)")
        model = "anthropic/claude-sonnet-4.5"
    elif anthropic_key:
        print(f"\nUsing Anthropic API (key: {anthropic_key[:20]}...)")
        model = "claude-sonnet-4-20250514"
    else:
        print("\nERROR: No API key found!")
        print("Set either OPENROUTER_API_KEY or ANTHROPIC_API_KEY environment variable.")
        return False

    # Create config with appropriate model
    config = APIConfig(model=model)
    print(f"Model: {model}")
    print("-" * 60)

    # Create client
    client = ClaudeClient(config)

    try:
        # Initialize
        print("\nInitializing client...")
        await client.initialize()
        print("Client initialized successfully!")

        # Send test message
        print("\nSending test message...")
        test_message = "Hello! Please respond with exactly: 'HARO voice assistant is working!'"

        response = await client.complete(
            user_input=test_message,
            system_prompt="You are HARO, a helpful voice assistant. Keep responses brief.",
            include_history=False,
        )

        print("\n" + "=" * 60)
        print("RESPONSE RECEIVED")
        print("=" * 60)
        print(f"\nText: {response.text}")
        print(f"\nModel: {response.model}")
        print(f"Latency: {response.latency:.2f}s")
        print(f"Usage: {response.usage}")
        print(f"Stop reason: {response.stop_reason}")

        # Get stats
        stats = client.get_stats()
        print(f"\nClient stats: {stats}")

        print("\n" + "=" * 60)
        print("TEST PASSED - API integration working!")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the async test."""
    parser = argparse.ArgumentParser(description="Test HARO API integration")
    parser.add_argument(
        "--key", "-k",
        help="API key (OpenRouter or Anthropic)",
        default=None
    )
    parser.add_argument(
        "--model", "-m",
        help="Model to use (default: auto-detect based on key type)",
        default=None
    )
    args = parser.parse_args()

    success = asyncio.run(test_api(api_key=args.key))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
