"""Lifecycle management for HARO.

Handles startup sequences, graceful shutdown, and component initialization.
"""

import asyncio
import signal
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, Awaitable, Any

from haro.core.config import HaroConfig
from haro.utils.logging import get_logger

logger = get_logger(__name__)


class LifecyclePhase(Enum):
    """Application lifecycle phases."""

    NOT_STARTED = auto()
    INITIALIZING = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class StartupResult:
    """Result of startup sequence."""

    success: bool
    phase: LifecyclePhase
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    initialized_components: list[str] = field(default_factory=list)


class LifecycleManager:
    """Manage HARO application lifecycle.

    Handles:
    - Ordered component initialization
    - Graceful shutdown with cleanup
    - Signal handling (SIGINT, SIGTERM)
    - Error recovery during startup
    """

    def __init__(self, config: HaroConfig) -> None:
        """Initialize lifecycle manager.

        Args:
            config: HARO configuration.
        """
        self.config = config
        self.phase = LifecyclePhase.NOT_STARTED
        self._components: dict[str, Any] = {}
        self._shutdown_handlers: list[Callable[[], Awaitable[None]]] = []
        self._signal_handlers_installed = False
        self._shutdown_event = asyncio.Event()

        self.logger = logger.bind(component="LifecycleManager")

    @property
    def is_running(self) -> bool:
        """Check if application is in running state."""
        return self.phase == LifecyclePhase.RUNNING

    def register_component(self, name: str, component: Any) -> None:
        """Register a component for lifecycle management.

        Args:
            name: Component name.
            component: Component instance.
        """
        self._components[name] = component
        self.logger.debug("component_registered", name=name)

    def register_shutdown_handler(
        self, handler: Callable[[], Awaitable[None]]
    ) -> None:
        """Register a shutdown handler.

        Args:
            handler: Async function to call during shutdown.
        """
        self._shutdown_handlers.append(handler)

    async def startup(self) -> StartupResult:
        """Execute startup sequence.

        Returns:
            StartupResult with initialization status.
        """
        result = StartupResult(
            success=True,
            phase=LifecyclePhase.INITIALIZING,
        )

        self.phase = LifecyclePhase.INITIALIZING
        self.logger.info("startup_beginning")

        # 1. Install signal handlers
        self._install_signal_handlers()

        # 2. Initialize logging (should already be done)
        result.initialized_components.append("logging")

        # 3. Load and validate configuration
        try:
            self._validate_config()
            result.initialized_components.append("config")
        except Exception as e:
            result.errors.append(f"Config validation failed: {e}")
            result.success = False
            self.phase = LifecyclePhase.ERROR
            return result

        # 4. Initialize audio subsystem
        try:
            await self._init_audio()
            result.initialized_components.append("audio")
        except Exception as e:
            result.warnings.append(f"Audio init warning: {e}")
            # Audio failures are warnings, not fatal

        # 5. Initialize speech components
        try:
            await self._init_speech()
            result.initialized_components.append("speech")
        except Exception as e:
            result.warnings.append(f"Speech init warning: {e}")

        # 6. Initialize context system
        try:
            await self._init_context()
            result.initialized_components.append("context")
        except Exception as e:
            result.warnings.append(f"Context init warning: {e}")

        # 7. Initialize API client
        try:
            await self._init_api()
            result.initialized_components.append("api")
        except Exception as e:
            result.warnings.append(f"API init warning: {e}")

        # Startup complete
        self.phase = LifecyclePhase.STARTING
        result.phase = LifecyclePhase.STARTING

        self.logger.info(
            "startup_complete",
            components=result.initialized_components,
            warnings=len(result.warnings),
        )

        return result

    async def run(self, agent) -> None:
        """Run the main application loop.

        Args:
            agent: The HaroAgent to run.
        """
        self.phase = LifecyclePhase.RUNNING
        self.logger.info("application_running")

        try:
            # Create agent run task
            agent_task = asyncio.create_task(agent.run())

            # Wait for shutdown signal or agent completion
            done, pending = await asyncio.wait(
                [agent_task, asyncio.create_task(self._shutdown_event.wait())],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        except asyncio.CancelledError:
            self.logger.info("application_cancelled")

        except Exception as e:
            self.logger.error("application_error", error=str(e))
            raise

    async def shutdown(self, reason: str = "requested") -> None:
        """Execute graceful shutdown.

        Args:
            reason: Reason for shutdown.
        """
        if self.phase == LifecyclePhase.STOPPING:
            return  # Already shutting down

        self.phase = LifecyclePhase.STOPPING
        self.logger.info("shutdown_beginning", reason=reason)

        # Signal shutdown
        self._shutdown_event.set()

        # Run registered shutdown handlers in reverse order
        for handler in reversed(self._shutdown_handlers):
            try:
                await asyncio.wait_for(handler(), timeout=5.0)
            except asyncio.TimeoutError:
                self.logger.warning("shutdown_handler_timeout")
            except Exception as e:
                self.logger.error("shutdown_handler_error", error=str(e))

        # Cleanup components
        await self._cleanup_components()

        self.phase = LifecyclePhase.STOPPED
        self.logger.info("shutdown_complete")

    def request_shutdown(self, reason: str = "signal") -> None:
        """Request application shutdown (thread-safe).

        Args:
            reason: Reason for shutdown.
        """
        self.logger.info("shutdown_requested", reason=reason)

        # Use call_soon_threadsafe for signal handlers
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self._shutdown_event.set)
        except RuntimeError:
            # No running loop, set directly
            self._shutdown_event.set()

    def _install_signal_handlers(self) -> None:
        """Install OS signal handlers for graceful shutdown."""
        if self._signal_handlers_installed:
            return

        def signal_handler(signum, frame):
            sig_name = signal.Signals(signum).name
            self.request_shutdown(reason=f"received {sig_name}")

        # Install handlers
        try:
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            self._signal_handlers_installed = True
            self.logger.debug("signal_handlers_installed")
        except Exception as e:
            self.logger.warning("signal_handler_install_failed", error=str(e))

    def _validate_config(self) -> None:
        """Validate configuration settings."""
        # Basic validation
        if self.config.audio.sample_rate < 8000:
            raise ValueError("Sample rate must be at least 8000 Hz")

        if self.config.api.timeout < 1:
            raise ValueError("API timeout must be at least 1 second")

    async def _init_audio(self) -> None:
        """Initialize audio components."""
        # Placeholder - actual implementation depends on components
        self.logger.debug("initializing_audio")

    async def _init_speech(self) -> None:
        """Initialize speech components (STT/TTS)."""
        # Placeholder - actual implementation depends on components
        self.logger.debug("initializing_speech")

    async def _init_context(self) -> None:
        """Initialize context management."""
        from haro.context import ContextManager

        if "context_manager" not in self._components:
            cm = ContextManager(self.config.context, self.config.api)
            cm.ensure_structure()
            self.register_component("context_manager", cm)

        self.logger.debug("context_initialized")

    async def _init_api(self) -> None:
        """Initialize API client."""
        # Placeholder - actual implementation depends on components
        self.logger.debug("initializing_api")

    async def _cleanup_components(self) -> None:
        """Clean up registered components."""
        for name, component in self._components.items():
            try:
                if hasattr(component, "close"):
                    await component.close()
                elif hasattr(component, "shutdown"):
                    await component.shutdown()
                elif hasattr(component, "stop"):
                    await component.stop()

                self.logger.debug("component_cleaned_up", name=name)

            except Exception as e:
                self.logger.error(
                    "component_cleanup_error",
                    name=name,
                    error=str(e),
                )


class StartupChecker:
    """Pre-flight checks before starting HARO.

    Verifies:
    - Required dependencies installed
    - Audio devices available
    - API key configured
    - Models downloaded
    """

    def __init__(self, config: HaroConfig) -> None:
        """Initialize startup checker.

        Args:
            config: HARO configuration.
        """
        self.config = config
        self.logger = logger.bind(component="StartupChecker")

    async def run_checks(self) -> dict:
        """Run all startup checks.

        Returns:
            Dictionary of check results.
        """
        results = {
            "all_passed": True,
            "checks": {},
        }

        # Check Python version
        results["checks"]["python_version"] = self._check_python()

        # Check dependencies
        results["checks"]["dependencies"] = self._check_dependencies()

        # Check audio
        results["checks"]["audio"] = await self._check_audio()

        # Check API key
        results["checks"]["api_key"] = self._check_api_key()

        # Check models
        results["checks"]["models"] = self._check_models()

        # Check context directory
        results["checks"]["context"] = self._check_context()

        # Determine overall status
        for check in results["checks"].values():
            if not check.get("passed", False) and check.get("required", True):
                results["all_passed"] = False
                break

        return results

    def _check_python(self) -> dict:
        """Check Python version."""
        import sys

        version = sys.version_info
        passed = version >= (3, 11)

        return {
            "passed": passed,
            "required": True,
            "message": f"Python {version.major}.{version.minor}.{version.micro}",
            "detail": "Requires Python 3.11+" if not passed else None,
        }

    def _check_dependencies(self) -> dict:
        """Check required dependencies."""
        required = ["numpy", "anthropic", "structlog", "yaml"]
        optional = ["sounddevice", "faster_whisper", "piper"]

        missing_required = []
        missing_optional = []

        for pkg in required:
            try:
                __import__(pkg)
            except ImportError:
                missing_required.append(pkg)

        for pkg in optional:
            try:
                __import__(pkg)
            except ImportError:
                missing_optional.append(pkg)

        passed = len(missing_required) == 0

        return {
            "passed": passed,
            "required": True,
            "message": f"Required: {len(required) - len(missing_required)}/{len(required)}, "
                       f"Optional: {len(optional) - len(missing_optional)}/{len(optional)}",
            "missing_required": missing_required,
            "missing_optional": missing_optional,
        }

    async def _check_audio(self) -> dict:
        """Check audio device availability."""
        try:
            import sounddevice as sd
            devices = sd.query_devices()

            input_devices = [d for d in devices if d["max_input_channels"] > 0]
            output_devices = [d for d in devices if d["max_output_channels"] > 0]

            has_input = len(input_devices) > 0
            has_output = len(output_devices) > 0

            return {
                "passed": has_input and has_output,
                "required": True,
                "message": f"Inputs: {len(input_devices)}, Outputs: {len(output_devices)}",
                "input_devices": len(input_devices),
                "output_devices": len(output_devices),
            }

        except ImportError:
            return {
                "passed": False,
                "required": True,
                "message": "sounddevice not installed",
            }

        except Exception as e:
            return {
                "passed": False,
                "required": True,
                "message": f"Audio check failed: {e}",
            }

    def _check_api_key(self) -> dict:
        """Check API key configuration."""
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        has_key = api_key is not None and len(api_key) > 0

        return {
            "passed": has_key,
            "required": False,  # Can run in offline mode
            "message": "API key configured" if has_key else "No API key found",
            "detail": "Set ANTHROPIC_API_KEY environment variable" if not has_key else None,
        }

    def _check_models(self) -> dict:
        """Check if models are available."""
        from pathlib import Path

        models_path = Path.home() / ".cache" / "haro" / "models"
        has_stt = False
        has_tts = False

        if models_path.exists():
            # Check for whisper model
            for f in models_path.glob("*whisper*"):
                has_stt = True
                break

            # Check for piper model
            for f in models_path.glob("*piper*"):
                has_tts = True
                break

        return {
            "passed": True,  # Models can be downloaded on first run
            "required": False,
            "message": f"STT: {'found' if has_stt else 'not found'}, "
                       f"TTS: {'found' if has_tts else 'not found'}",
            "has_stt": has_stt,
            "has_tts": has_tts,
        }

    def _check_context(self) -> dict:
        """Check context directory."""
        from pathlib import Path

        context_path = Path(self.config.context.path).expanduser()
        exists = context_path.exists()

        return {
            "passed": True,  # Will be created if needed
            "required": False,
            "message": "Context directory exists" if exists else "Will be created on startup",
            "path": str(context_path),
        }
