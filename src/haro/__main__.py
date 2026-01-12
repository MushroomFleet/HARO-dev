"""HARO CLI entry point.

Provides command-line interface for HARO voice assistant.
Run with: python -m haro
"""

import asyncio
import sys
from typing import Optional

import click
from rich.console import Console
from rich.table import Table

from haro import __version__
from haro.core.config import load_config, HaroConfig
from haro.utils.logging import setup_logging, get_logger

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="HARO")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    help="Path to configuration file",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging",
)
@click.pass_context
def cli(ctx: click.Context, config: Optional[str], debug: bool) -> None:
    """HARO - Helpful Autonomous Responsive Operator

    A lightweight, always-listening voice AI assistant.
    """
    ctx.ensure_object(dict)

    # Load configuration
    haro_config = load_config(config)
    ctx.obj["config"] = haro_config

    # Set up logging
    log_level = "DEBUG" if debug else haro_config.logging.level
    setup_logging(
        level=log_level,
        log_file=haro_config.logging.file,
        console=haro_config.logging.console,
    )


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Check system status and configuration."""
    config: HaroConfig = ctx.obj["config"]

    console.print("\n[bold blue]HARO Status[/bold blue]\n")

    # Version info
    console.print(f"[bold]Version:[/bold] {__version__}")
    console.print(f"[bold]Device ID:[/bold] {config.device_id}")
    console.print()

    # Check dependencies
    console.print("[bold]Dependencies:[/bold]")
    deps_table = Table(show_header=False, box=None)
    deps_table.add_column("Package", style="cyan")
    deps_table.add_column("Status")

    dependencies = [
        ("sounddevice", "sounddevice"),
        ("numpy", "numpy"),
        ("structlog", "structlog"),
        ("PyYAML", "yaml"),
        ("click", "click"),
        ("rich", "rich"),
        ("anthropic", "anthropic"),
    ]

    for name, module in dependencies:
        try:
            __import__(module)
            deps_table.add_row(name, "[green]OK[/green]")
        except ImportError:
            deps_table.add_row(name, "[red]Missing[/red]")

    console.print(deps_table)
    console.print()

    # Audio devices
    console.print("[bold]Audio Devices:[/bold]")
    try:
        import sounddevice as sd

        # Input devices
        console.print("  [cyan]Input devices:[/cyan]")
        devices = sd.query_devices()
        default_input = sd.default.device[0]

        input_table = Table(show_header=True, box=None, padding=(0, 2))
        input_table.add_column("ID", style="dim")
        input_table.add_column("Name")
        input_table.add_column("Channels")
        input_table.add_column("Default")

        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                is_default = "*" if i == default_input else ""
                input_table.add_row(
                    str(i),
                    dev["name"][:40],
                    str(dev["max_input_channels"]),
                    is_default,
                )

        console.print(input_table)

        # Output devices
        console.print("  [cyan]Output devices:[/cyan]")
        default_output = sd.default.device[1]

        output_table = Table(show_header=True, box=None, padding=(0, 2))
        output_table.add_column("ID", style="dim")
        output_table.add_column("Name")
        output_table.add_column("Channels")
        output_table.add_column("Default")

        for i, dev in enumerate(devices):
            if dev["max_output_channels"] > 0:
                is_default = "*" if i == default_output else ""
                output_table.add_row(
                    str(i),
                    dev["name"][:40],
                    str(dev["max_output_channels"]),
                    is_default,
                )

        console.print(output_table)

    except Exception as e:
        console.print(f"  [red]Error listing devices: {e}[/red]")

    console.print()

    # Configuration summary
    console.print("[bold]Configuration:[/bold]")
    config_table = Table(show_header=False, box=None)
    config_table.add_column("Setting", style="cyan")
    config_table.add_column("Value")

    config_table.add_row("Sample Rate", f"{config.audio.sample_rate} Hz")
    config_table.add_row("Channels", str(config.audio.channels))
    config_table.add_row("Chunk Size", str(config.audio.chunk_size))
    config_table.add_row("Wake Phrase", f'"{config.wake.phrase}"')
    config_table.add_row("Wake Sensitivity", str(config.wake.sensitivity))
    config_table.add_row("VAD Threshold", str(config.vad.threshold))
    config_table.add_row("STT Model", config.stt.model)
    config_table.add_row("TTS Engine", config.tts.engine)
    config_table.add_row("API Model", config.api.model)
    config_table.add_row("Context Path", config.context.path)

    console.print(config_table)
    console.print()

    # Check .context directory
    from pathlib import Path

    context_path = Path(config.context.path)
    if context_path.exists():
        console.print("[bold].context Directory:[/bold] [green]Found[/green]")
        for subdir in ["sessions", "knowledge", "config"]:
            subpath = context_path / subdir
            status = "[green]OK[/green]" if subpath.exists() else "[yellow]Missing[/yellow]"
            console.print(f"  {subdir}/: {status}")
    else:
        console.print(
            "[bold].context Directory:[/bold] [yellow]Not initialized[/yellow]"
        )
        console.print("  Run 'haro init-context' to create")

    console.print()


@cli.command("init-context")
@click.option(
    "--path",
    "-p",
    type=click.Path(),
    help="Path for .context directory (default: .context/)",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing files",
)
@click.pass_context
def init_context_cmd(
    ctx: click.Context, path: Optional[str], overwrite: bool
) -> None:
    """Initialize the .context directory structure."""
    from haro.context import init_context

    config: HaroConfig = ctx.obj["config"]
    context_path = path or config.context.path

    console.print(f"\n[bold]Initializing .context at:[/bold] {context_path}\n")

    result = init_context(path=context_path, overwrite=overwrite)

    if result.created_dirs:
        console.print("[bold]Created directories:[/bold]")
        for d in result.created_dirs:
            console.print(f"  [green]+[/green] {d}/")

    if result.created_files:
        console.print("[bold]Created files:[/bold]")
        for f in result.created_files:
            console.print(f"  [green]+[/green] {f}")

    if result.skipped_files:
        console.print("[bold]Skipped (already exist):[/bold]")
        for f in result.skipped_files:
            console.print(f"  [yellow]-[/yellow] {f}")

    if result.errors:
        console.print("[bold red]Errors:[/bold red]")
        for e in result.errors:
            console.print(f"  [red]![/red] {e}")

    console.print()

    if result.success:
        console.print("[green]Context initialization complete![/green]\n")
    else:
        console.print("[red]Context initialization failed.[/red]\n")
        sys.exit(1)


@cli.command()
@click.option(
    "--parallel",
    is_flag=True,
    help="Use parallel worker architecture for faster response",
)
@click.pass_context
def run(ctx: click.Context, parallel: bool) -> None:
    """Run the HARO voice assistant."""
    config: HaroConfig = ctx.obj["config"]
    logger = get_logger(__name__)

    mode_str = "[yellow]parallel[/yellow]" if parallel else "sequential"
    console.print(f"\n[bold blue]Starting HARO ({mode_str} mode)...[/bold blue]\n")
    console.print(f"Wake phrase: [cyan]\"{config.wake.phrase}\"[/cyan]")
    console.print(f"Sensitivity: [cyan]{config.wake.sensitivity}[/cyan]")
    console.print("Press Ctrl+C to stop\n")

    async def main() -> None:
        if parallel:
            from haro.core.parallel_agent import ParallelAgent as AgentClass
        else:
            from haro.core.agent import HaroAgent as AgentClass
        from haro.audio import AudioCapture, AudioPlayback, VoiceActivityDetector, AudioFeedback
        from haro.audio.wake import WakeWordDetector
        from haro.speech import WhisperSTT, PiperTTS
        from haro.intelligence import ClaudeClient, PromptBuilder, ResponseParser

        logger.info(
            "haro_starting",
            version=__version__,
            device_id=config.device_id,
        )

        # Initialize components
        console.print("[cyan]Initializing components...[/cyan]")

        # Audio
        capture = AudioCapture(config.audio)
        playback = AudioPlayback(config.audio)
        vad = VoiceActivityDetector(config.vad)
        feedback = AudioFeedback(playback, config.wake)

        # Speech
        console.print(f"  Loading STT model ({config.stt.model})...")
        stt = WhisperSTT(config.stt)
        try:
            await stt.load_model()
            console.print("  [green]STT model loaded[/green]")
        except Exception as e:
            console.print(f"  [red]Failed to load STT model: {e}[/red]")
            console.print("  [yellow]Run: haro download-model tiny.en[/yellow]")
            return

        # TTS
        console.print(f"  Loading TTS voice ({config.tts.voice})...")
        tts = PiperTTS(config.tts)
        try:
            await tts.load_voice()
            console.print("  [green]TTS voice loaded[/green]")
            # Wire up TTS to feedback for verbal confirmations
            await feedback.set_tts(tts)
            console.print("  [green]TTS connected to feedback system[/green]")
        except Exception as e:
            console.print(f"  [yellow]TTS not available: {e}[/yellow]")
            tts = None

        # Wake word detector
        wake_detector = WakeWordDetector(config.wake, config.audio)

        # Intelligence components
        console.print("  Initializing Claude API client...")
        api_client = ClaudeClient(config.api)
        prompt_builder = PromptBuilder(config.context, config.wake)
        response_parser = ResponseParser()
        try:
            await api_client.initialize()
            console.print("  [green]Claude API client ready[/green]")
        except Exception as e:
            console.print(f"  [yellow]API not available: {e}[/yellow]")
            console.print("  [yellow]HARO will echo responses without API[/yellow]")
            api_client = None

        # Context manager
        from haro.context import ContextManager
        context_manager = ContextManager(config.context, config.api)
        context_manager.ensure_structure()
        context_manager.start_session(device_id=config.device_id)
        console.print("  [green]Context manager ready[/green]")

        # Create agent
        agent = AgentClass(config)
        await agent.initialize(
            audio_capture=capture,
            audio_playback=playback,
            wake_detector=wake_detector,
            vad=vad,
            stt=stt,
            tts=tts,
            feedback=feedback,
            api_client=api_client,
            prompt_builder=prompt_builder,
            response_parser=response_parser,
            context_manager=context_manager,
        )

        console.print("\n[bold green]HARO is ready![/bold green]")
        console.print(f"Say \"[cyan]{config.wake.phrase}[/cyan]\" to activate.\n")

        # Start playback before playing ready sound
        await playback.start()

        # Play ready sound
        await feedback.play_ready()

        # Run agent
        logger.info("about_to_start_agent_run")
        try:
            await agent.run()
        except Exception as e:
            logger.error("agent_run_exception", error=str(e), exc_info=True)
            raise
        finally:
            # End session and save
            context_manager.end_session()
            logger.info("haro_stopped")

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[bold]HARO stopped.[/bold]\n")


@cli.command("test-audio")
@click.option(
    "--duration",
    "-d",
    type=float,
    default=3.0,
    help="Duration of test in seconds",
)
@click.pass_context
def test_audio(ctx: click.Context, duration: float) -> None:
    """Test audio capture and playback."""
    config: HaroConfig = ctx.obj["config"]

    console.print("\n[bold]Testing audio...[/bold]\n")

    async def run_test() -> None:
        from haro.audio import AudioCapture, AudioPlayback
        import numpy as np

        # Test capture
        console.print("[cyan]Testing audio capture...[/cyan]")
        capture = AudioCapture(config.audio)

        try:
            await capture.start()
            console.print(f"  Recording for {duration} seconds...")

            chunks = []
            elapsed = 0.0
            chunk_duration = config.audio.chunk_size / config.audio.sample_rate

            while elapsed < duration:
                chunk = await capture.read_chunk(timeout=1.0)
                if chunk is not None:
                    chunks.append(chunk)
                    elapsed += chunk_duration

            await capture.stop()

            if chunks:
                audio = np.concatenate(chunks)
                energy = np.sqrt(np.mean(audio**2))
                console.print(f"  [green]Captured {len(audio)} samples[/green]")
                console.print(f"  Average energy: {energy:.4f}")
            else:
                console.print("  [red]No audio captured[/red]")

        except Exception as e:
            console.print(f"  [red]Capture error: {e}[/red]")
            return

        # Test playback
        console.print("\n[cyan]Testing audio playback...[/cyan]")
        playback = AudioPlayback(config.audio)

        try:
            await playback.start()
            console.print("  Playing test tone (440 Hz)...")
            await playback.play_tone(frequency=440.0, duration=0.5, wait=True)
            await playback.stop()
            console.print("  [green]Playback complete[/green]")
        except Exception as e:
            console.print(f"  [red]Playback error: {e}[/red]")
            return

        console.print("\n[green]Audio test complete![/green]\n")

    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        console.print("\n[bold]Test cancelled.[/bold]\n")


@cli.command("models")
@click.pass_context
def list_models(ctx: click.Context) -> None:
    """List available and installed speech models."""
    from haro.speech import ModelManager, WhisperSTT, PiperTTS

    config: HaroConfig = ctx.obj["config"]
    manager = ModelManager(base_path=config.stt.model_path)

    console.print("\n[bold blue]Speech Models[/bold blue]\n")

    # List Whisper models
    console.print("[bold]STT Models (Whisper):[/bold]")
    stt_table = Table(show_header=True, box=None, padding=(0, 2))
    stt_table.add_column("Model", style="cyan")
    stt_table.add_column("Size")
    stt_table.add_column("Params")
    stt_table.add_column("Status")

    for model_name in WhisperSTT.get_available_models():
        info = WhisperSTT.get_model_info(model_name)
        is_installed = manager.is_model_installed(model_name)
        status = "[green]Installed[/green]" if is_installed else "[dim]Not installed[/dim]"
        current = " [yellow]*[/yellow]" if model_name == config.stt.model else ""

        stt_table.add_row(
            f"{model_name}{current}",
            f"{info.get('size_mb', '?')} MB",
            info.get("parameters", "?"),
            status,
        )

    console.print(stt_table)
    console.print()

    # List Piper voices
    console.print("[bold]TTS Voices (Piper):[/bold]")
    tts_table = Table(show_header=True, box=None, padding=(0, 2))
    tts_table.add_column("Voice", style="cyan")
    tts_table.add_column("Size")
    tts_table.add_column("Quality")
    tts_table.add_column("Status")

    for voice_name in PiperTTS.get_available_voices():
        info = PiperTTS.get_voice_info(voice_name)
        is_installed = manager.is_model_installed(voice_name)
        status = "[green]Installed[/green]" if is_installed else "[dim]Not installed[/dim]"
        current = " [yellow]*[/yellow]" if voice_name == config.tts.voice else ""

        tts_table.add_row(
            f"{voice_name}{current}",
            f"{info.get('size_mb', '?')} MB",
            info.get("quality", "?"),
            status,
        )

    console.print(tts_table)
    console.print()

    # Show total storage used
    total_mb = manager.get_total_size() / (1024 * 1024)
    console.print(f"[bold]Total storage used:[/bold] {total_mb:.1f} MB\n")


@cli.command("download-model")
@click.argument("model_name")
@click.pass_context
def download_model(ctx: click.Context, model_name: str) -> None:
    """Download a speech model.

    MODEL_NAME: Name of the model to download (e.g., tiny.en, en_US-lessac-medium)
    """
    from haro.speech import ModelManager, WhisperSTT, PiperTTS
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

    config: HaroConfig = ctx.obj["config"]
    manager = ModelManager(base_path=config.stt.model_path)

    # Determine model type
    is_stt = model_name in WhisperSTT.get_available_models()
    is_tts = model_name in PiperTTS.get_available_voices()

    if not is_stt and not is_tts:
        console.print(f"[red]Unknown model: {model_name}[/red]")
        console.print("Use 'haro models' to see available models.")
        sys.exit(1)

    if manager.is_model_installed(model_name):
        console.print(f"[yellow]Model '{model_name}' is already installed.[/yellow]")
        return

    console.print(f"\n[bold]Downloading {model_name}...[/bold]\n")

    async def do_download() -> None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(f"Downloading {model_name}...", total=100)

            def update_progress(p):
                progress.update(task, completed=p.percent)

            try:
                if is_stt:
                    await manager.download_whisper_model(
                        model_name, progress_callback=update_progress
                    )
                else:
                    await manager.download_piper_voice(
                        model_name, progress_callback=update_progress
                    )

                progress.update(task, completed=100)
                console.print(f"\n[green]Successfully downloaded {model_name}![/green]\n")

            except Exception as e:
                console.print(f"\n[red]Download failed: {e}[/red]\n")
                sys.exit(1)

    asyncio.run(do_download())


@cli.command("test-stt")
@click.option(
    "--duration",
    "-d",
    type=float,
    default=5.0,
    help="Recording duration in seconds",
)
@click.pass_context
def test_stt(ctx: click.Context, duration: float) -> None:
    """Test speech-to-text transcription."""
    config: HaroConfig = ctx.obj["config"]

    console.print("\n[bold]Testing Speech-to-Text...[/bold]\n")

    async def run_test() -> None:
        from haro.audio import AudioCapture
        from haro.speech import WhisperSTT
        import numpy as np

        # Initialize STT
        console.print(f"[cyan]Loading Whisper model ({config.stt.model})...[/cyan]")
        stt = WhisperSTT(config.stt)

        try:
            await stt.load_model()
            console.print("  [green]Model loaded[/green]")
        except Exception as e:
            console.print(f"  [red]Failed to load model: {e}[/red]")
            console.print("  Try: haro download-model tiny.en")
            return

        # Capture audio
        console.print(f"\n[cyan]Recording for {duration} seconds...[/cyan]")
        console.print("  Speak now!")

        capture = AudioCapture(config.audio)
        await capture.start()

        chunks = []
        elapsed = 0.0
        chunk_duration = config.audio.chunk_size / config.audio.sample_rate

        while elapsed < duration:
            chunk = await capture.read_chunk(timeout=1.0)
            if chunk is not None:
                chunks.append(chunk)
                elapsed += chunk_duration

        await capture.stop()

        if not chunks:
            console.print("  [red]No audio captured[/red]")
            return

        audio = np.concatenate(chunks)
        console.print(f"  [green]Captured {len(audio)} samples ({elapsed:.1f}s)[/green]")

        # Transcribe
        console.print("\n[cyan]Transcribing...[/cyan]")
        try:
            result = await stt.transcribe(audio, sample_rate=config.audio.sample_rate)
            console.print(f"\n[bold]Transcription:[/bold]")
            console.print(f"  \"{result.text}\"")
            console.print(f"\n  Language: {result.language}")
            console.print(f"  Confidence: {result.confidence:.2f}")
        except Exception as e:
            console.print(f"  [red]Transcription failed: {e}[/red]")

    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        console.print("\n[bold]Test cancelled.[/bold]\n")


@cli.command("test-wake")
@click.option(
    "--duration",
    "-d",
    type=float,
    default=30.0,
    help="Listening duration in seconds",
)
@click.option(
    "--sensitivity",
    "-s",
    type=float,
    default=None,
    help="Detection sensitivity (0.0-1.0)",
)
@click.pass_context
def test_wake(ctx: click.Context, duration: float, sensitivity: float) -> None:
    """Test wake word detection.

    Listens for the wake word and shows detection results.
    Press Ctrl+C to stop.
    """
    config: HaroConfig = ctx.obj["config"]

    # Override sensitivity if provided
    if sensitivity is not None:
        config.wake.sensitivity = sensitivity

    console.print("\n[bold]Testing Wake Word Detection...[/bold]\n")
    console.print(f"Wake phrase: [cyan]\"{config.wake.phrase}\"[/cyan]")
    console.print(f"Sensitivity: [cyan]{config.wake.sensitivity}[/cyan]")
    console.print(f"Duration: [cyan]{duration}s[/cyan]")
    console.print("\nSay the wake word to trigger detection...")
    console.print("Press Ctrl+C to stop\n")

    async def run_test() -> None:
        from haro.audio import AudioCapture, AudioPlayback, AudioFeedback
        from haro.audio.wake import WakeWordDetector
        from haro.speech import WhisperSTT
        import numpy as np
        import time

        # Initialize STT
        console.print("[cyan]Loading Whisper model...[/cyan]")
        stt = WhisperSTT(config.stt)

        try:
            await stt.load_model()
            console.print("  [green]Model loaded[/green]")
        except Exception as e:
            console.print(f"  [red]Failed to load model: {e}[/red]")
            console.print("  Try: haro download-model tiny.en")
            return

        # Initialize wake detector
        detector = WakeWordDetector(config.wake, config.audio)
        await detector.initialize(stt)
        console.print("  [green]Wake detector initialized[/green]")

        # Initialize audio feedback
        playback = AudioPlayback(config.audio)
        await playback.start()
        feedback = AudioFeedback(playback, config.wake)
        console.print("  [green]Audio feedback ready[/green]")

        # Initialize capture
        capture = AudioCapture(config.audio)
        await capture.start()
        console.print("  [green]Audio capture started[/green]\n")

        console.print("[bold green]Listening...[/bold green]\n")

        start_time = time.time()
        detection_count = 0
        check_interval = 0.5  # Check for wake word every 0.5 seconds
        last_check = 0.0

        try:
            while time.time() - start_time < duration:
                # Read audio chunk
                chunk = await capture.read_chunk(timeout=0.1)
                if chunk is not None:
                    detector.add_audio(chunk)

                # Periodic detection check
                elapsed = time.time() - start_time
                if elapsed - last_check >= check_interval:
                    last_check = elapsed

                    # Check for wake word
                    result = await detector.detect()

                    if result.detected:
                        detection_count += 1
                        console.print(
                            f"[bold green]✓ WAKE WORD DETECTED![/bold green] "
                            f"(confidence: {result.confidence:.2f}, "
                            f"text: \"{result.text}\")"
                        )

                        # Play confirmation
                        await feedback.play_wake_confirmation(use_verbal=False)

                        # Clear buffer to prevent re-detecting same audio
                        # Natural 2s cooldown prevents immediate re-detection
                        detector.clear_buffer()

        except asyncio.CancelledError:
            pass

        await capture.stop()
        await playback.stop()

        console.print(f"\n[bold]Test complete![/bold]")
        console.print(f"  Duration: {time.time() - start_time:.1f}s")
        console.print(f"  Detections: {detection_count}")
        console.print()

    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        console.print("\n[bold]Test stopped.[/bold]\n")


@cli.command("test-tts")
@click.argument("text", default="Hello! I am HARO, your voice assistant.")
@click.pass_context
def test_tts(ctx: click.Context, text: str) -> None:
    """Test text-to-speech synthesis.

    TEXT: Text to synthesize (default: greeting)
    """
    config: HaroConfig = ctx.obj["config"]

    console.print("\n[bold]Testing Text-to-Speech...[/bold]\n")

    async def run_test() -> None:
        from haro.audio import AudioPlayback
        from haro.speech import PiperTTS

        # Initialize TTS
        console.print(f"[cyan]Loading Piper voice ({config.tts.voice})...[/cyan]")
        tts = PiperTTS(config.tts)
        playback = None
        result = None

        try:
            await tts.load_voice()
            console.print("  [green]Voice loaded[/green]")

            # Synthesize
            console.print(f"\n[cyan]Synthesizing:[/cyan] \"{text}\"")
            result = await tts.synthesize(text)
            console.print(f"  [green]Generated {result.duration:.2f}s of audio[/green]")

            # Play
            console.print("\n[cyan]Playing audio...[/cyan]")
            playback = AudioPlayback(config.audio)
            await playback.start()

            try:
                await playback.play(
                    result.audio,
                    sample_rate=result.sample_rate,
                    wait=True,
                )
                console.print("  [green]Playback complete[/green]")
            except Exception as e:
                console.print(f"  [red]Playback failed: {e}[/red]")
            finally:
                await playback.stop()

        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            if not tts.is_loaded:
                console.print(f"  Try: haro download-model {config.tts.voice}")
        finally:
            # CRITICAL: Always unload TTS to prevent GPU memory leak
            await tts.unload_voice()

    try:
        asyncio.run(run_test())
    except KeyboardInterrupt:
        console.print("\n[bold]Test cancelled.[/bold]\n")


def main() -> None:
    """Main entry point."""
    cli(obj={})


if __name__ == "__main__":
    main()
