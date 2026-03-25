# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""tritium_lib.pipeline — configurable data pipeline orchestrator.

Chains sensor ingestion, tracking, fusion, alerting, and reporting into a
single managed flow with backpressure management and lifecycle control.

Architecture
------------
- **PipelineStage** — abstract base class for a processing stage (input -> output)
- **Pipeline** — configurable chain of processing stages
- **PipelineConfig** — dict/YAML-friendly configuration for pipeline topology
- **PipelineRunner** — runs the pipeline with backpressure management

Built-in stages:
  - IngestStage   — receives raw sensor data and normalizes it
  - TrackingStage — updates the target tracker with ingested data
  - FusionStage   — runs multi-sensor correlation via FusionEngine
  - AlertingStage — evaluates alert rules against pipeline data
  - ReportingStage — generates periodic situation reports

Quick start::

    from tritium_lib.pipeline import Pipeline, IngestStage, TrackingStage
    from tritium_lib.pipeline import FusionStage, AlertingStage, ReportingStage
    from tritium_lib.tracking import TargetTracker
    from tritium_lib.fusion import FusionEngine
    from tritium_lib.alerting import AlertEngine

    pipeline = Pipeline([
        IngestStage(sources=["ble", "wifi", "camera"]),
        TrackingStage(tracker=TargetTracker()),
        FusionStage(engine=FusionEngine()),
        AlertingStage(rules=AlertEngine().get_rules()),
        ReportingStage(interval=3600),
    ])
    pipeline.start()

    # Push data through
    pipeline.push({"source": "ble", "mac": "AA:BB:CC:DD:EE:FF", "rssi": -55})
    pipeline.push({"source": "camera", "class_name": "person", "confidence": 0.9})

    pipeline.stop()
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Pipeline data envelope
# ---------------------------------------------------------------------------

@dataclass
class PipelineMessage:
    """A message flowing through the pipeline.

    Wraps the raw data with metadata for routing, tracing, and timing.
    """
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    data: dict = field(default_factory=dict)
    source: str = ""
    stage_trace: list[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    errors: list[str] = field(default_factory=list)
    dropped: bool = False

    def clone(self) -> "PipelineMessage":
        """Create a shallow copy with a new message ID."""
        return PipelineMessage(
            message_id=uuid.uuid4().hex[:12],
            data=dict(self.data),
            source=self.source,
            stage_trace=list(self.stage_trace),
            timestamp=self.timestamp,
            errors=list(self.errors),
            dropped=self.dropped,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "data": self.data,
            "source": self.source,
            "stage_trace": self.stage_trace,
            "timestamp": self.timestamp,
            "errors": self.errors,
            "dropped": self.dropped,
        }


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

class PipelineState(str, Enum):
    """Pipeline lifecycle states."""
    CREATED = "created"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


# ---------------------------------------------------------------------------
# PipelineStage — abstract base
# ---------------------------------------------------------------------------

class PipelineStage(ABC):
    """Abstract base class for a pipeline processing stage.

    Each stage receives a PipelineMessage, processes it, and returns
    zero or more output messages. A stage can filter (drop), transform,
    enrich, or fan-out messages.

    Subclasses must implement ``process()``. Optionally override
    ``setup()`` and ``teardown()`` for lifecycle management.
    """

    def __init__(self, name: str = "") -> None:
        self._name = name or self.__class__.__name__
        self._processed_count: int = 0
        self._error_count: int = 0
        self._dropped_count: int = 0
        self._total_time_ms: float = 0.0

    @property
    def name(self) -> str:
        return self._name

    @property
    def stats(self) -> dict[str, Any]:
        """Per-stage statistics."""
        return {
            "name": self._name,
            "processed": self._processed_count,
            "errors": self._error_count,
            "dropped": self._dropped_count,
            "total_time_ms": round(self._total_time_ms, 2),
            "avg_time_ms": round(
                self._total_time_ms / max(self._processed_count, 1), 2
            ),
        }

    def setup(self) -> None:
        """Called once when the pipeline starts. Override for initialization."""
        pass

    def teardown(self) -> None:
        """Called once when the pipeline stops. Override for cleanup."""
        pass

    @abstractmethod
    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        """Process a message and return output messages.

        Parameters
        ----------
        message:
            The incoming pipeline message.

        Returns
        -------
        list[PipelineMessage]:
            Zero or more output messages. Return an empty list to drop
            the message. Return a list with the (possibly modified)
            input message to pass it through. Return multiple messages
            to fan out.
        """
        ...

    def _run(self, message: PipelineMessage) -> list[PipelineMessage]:
        """Internal runner that wraps process() with timing and error handling."""
        start = time.monotonic()
        try:
            results = self.process(message)
            self._processed_count += 1
            for msg in results:
                msg.stage_trace.append(self._name)
            if not results:
                self._dropped_count += 1
            return results
        except Exception as exc:
            self._error_count += 1
            message.errors.append(f"{self._name}: {exc}")
            logger.warning("Stage %s error: %s", self._name, exc)
            return [message]
        finally:
            elapsed = (time.monotonic() - start) * 1000
            self._total_time_ms += elapsed


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    """Configuration for pipeline topology.

    Can be constructed from a plain dict for YAML/JSON config file support.

    Attributes
    ----------
    name:
        Human-readable pipeline name.
    stages:
        Ordered list of stage configuration dicts. Each dict must have
        a ``type`` key matching a registered stage factory.
    max_queue_size:
        Maximum backpressure queue depth (0 = unbounded).
    error_policy:
        What to do on stage errors: "propagate" (pass through with error),
        "drop" (discard message), "halt" (stop the pipeline).
    """
    name: str = "default"
    stages: list[dict[str, Any]] = field(default_factory=list)
    max_queue_size: int = 10000
    error_policy: str = "propagate"

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PipelineConfig":
        """Construct a PipelineConfig from a plain dictionary."""
        return cls(
            name=d.get("name", "default"),
            stages=d.get("stages", []),
            max_queue_size=d.get("max_queue_size", 10000),
            error_policy=d.get("error_policy", "propagate"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stages": self.stages,
            "max_queue_size": self.max_queue_size,
            "error_policy": self.error_policy,
        }


# ---------------------------------------------------------------------------
# Built-in stage: IngestStage
# ---------------------------------------------------------------------------

VALID_SOURCES = {"ble", "wifi", "camera", "acoustic", "mesh", "adsb", "rf_motion"}


class IngestStage(PipelineStage):
    """Receives raw sensor data, validates the source, and normalizes it.

    Filters messages whose ``source`` field is not in the allowed set.
    Adds a normalized timestamp if missing.

    Parameters
    ----------
    sources:
        List of accepted source types. Defaults to all known sources.
    """

    def __init__(self, sources: list[str] | None = None, name: str = "") -> None:
        super().__init__(name=name or "IngestStage")
        self._sources = set(sources) if sources else set(VALID_SOURCES)

    @property
    def sources(self) -> set[str]:
        return set(self._sources)

    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        source = message.data.get("source", message.source)
        if not source:
            # Try to infer source from data keys
            if "mac" in message.data and "rssi" in message.data:
                source = "ble"
            elif "ssid" in message.data:
                source = "wifi"
            elif "class_name" in message.data:
                source = "camera"
            elif "event_type" in message.data:
                source = "acoustic"
            elif "target_id" in message.data:
                d = message.data
                if d.get("target_id", "").startswith("mesh_"):
                    source = "mesh"
                elif d.get("target_id", "").startswith("adsb_"):
                    source = "adsb"

        if source not in self._sources:
            logger.debug("IngestStage dropping unknown source: %s", source)
            return []

        # Normalize: ensure source is set on both message and data
        message.source = source
        message.data["source"] = source
        if "timestamp" not in message.data:
            message.data["timestamp"] = time.time()

        return [message]


# ---------------------------------------------------------------------------
# Built-in stage: TrackingStage
# ---------------------------------------------------------------------------

class TrackingStage(PipelineStage):
    """Updates the target tracker with ingested sensor data.

    Dispatches to the appropriate TargetTracker.update_from_*() method
    based on the message source type.

    Parameters
    ----------
    tracker:
        A TargetTracker instance to update. If None, a new one is created.
    """

    def __init__(self, tracker: Any = None, name: str = "") -> None:
        super().__init__(name=name or "TrackingStage")
        self._tracker = tracker

    def setup(self) -> None:
        if self._tracker is None:
            from tritium_lib.tracking import TargetTracker
            self._tracker = TargetTracker()

    @property
    def tracker(self) -> Any:
        return self._tracker

    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        if self._tracker is None:
            return [message]

        source = message.source or message.data.get("source", "")
        data = message.data

        _SOURCE_METHOD_MAP = {
            "ble": "update_from_ble",
            "wifi": "update_from_ble",  # WiFi probes go through BLE path
            "camera": "update_from_detection",
            "acoustic": "update_from_rf_motion",
            "mesh": "update_from_mesh",
            "adsb": "update_from_adsb",
            "rf_motion": "update_from_rf_motion",
        }

        method_name = _SOURCE_METHOD_MAP.get(source)
        if method_name:
            method = getattr(self._tracker, method_name, None)
            if method:
                try:
                    method(data)
                    message.data["tracked"] = True
                except Exception as exc:
                    message.errors.append(f"TrackingStage: {exc}")

        return [message]


# ---------------------------------------------------------------------------
# Built-in stage: FusionStage
# ---------------------------------------------------------------------------

class FusionStage(PipelineStage):
    """Runs multi-sensor fusion and correlation.

    Ingests data into the FusionEngine and optionally triggers a
    correlation pass after each batch.

    Parameters
    ----------
    engine:
        A FusionEngine instance. If None, a new one is created on setup.
    auto_correlate:
        If True, run correlation after every message. Default False
        (correlation runs on a timer or manually).
    """

    def __init__(
        self,
        engine: Any = None,
        *,
        auto_correlate: bool = False,
        name: str = "",
    ) -> None:
        super().__init__(name=name or "FusionStage")
        self._engine = engine
        self._auto_correlate = auto_correlate

    def setup(self) -> None:
        if self._engine is None:
            from tritium_lib.fusion import FusionEngine
            self._engine = FusionEngine()

    @property
    def engine(self) -> Any:
        return self._engine

    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        if self._engine is None:
            return [message]

        source = message.source or message.data.get("source", "")
        data = message.data

        _SOURCE_INGEST_MAP = {
            "ble": "ingest_ble",
            "wifi": "ingest_wifi",
            "camera": "ingest_camera",
            "acoustic": "ingest_acoustic",
            "mesh": "ingest_mesh",
            "adsb": "ingest_adsb",
            "rf_motion": "ingest_rf_motion",
        }

        method_name = _SOURCE_INGEST_MAP.get(source)
        if method_name:
            method = getattr(self._engine, method_name, None)
            if method:
                try:
                    target_id = method(data)
                    if target_id:
                        message.data["fused_target_id"] = target_id
                except Exception as exc:
                    message.errors.append(f"FusionStage: {exc}")

        if self._auto_correlate:
            try:
                correlations = self._engine.run_correlation()
                if correlations:
                    message.data["new_correlations"] = len(correlations)
            except Exception as exc:
                message.errors.append(f"FusionStage correlation: {exc}")

        return [message]


# ---------------------------------------------------------------------------
# Built-in stage: AlertingStage
# ---------------------------------------------------------------------------

class AlertingStage(PipelineStage):
    """Evaluates alert rules against pipeline data.

    Takes the processed message and evaluates it against the alert
    engine's rule set. Fired alerts are attached to the message.

    Parameters
    ----------
    alert_engine:
        An AlertEngine instance. If None, a new one is created on setup.
    rules:
        Optional list of AlertRule objects to load into the engine.
    """

    def __init__(
        self,
        alert_engine: Any = None,
        *,
        rules: list | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name or "AlertingStage")
        self._alert_engine = alert_engine
        self._initial_rules = rules or []
        self._alerts_fired: int = 0

    def setup(self) -> None:
        if self._alert_engine is None:
            from tritium_lib.alerting import AlertEngine
            self._alert_engine = AlertEngine(load_defaults=True)
        for rule in self._initial_rules:
            self._alert_engine.add_rule(rule)

    @property
    def alert_engine(self) -> Any:
        return self._alert_engine

    @property
    def alerts_fired(self) -> int:
        return self._alerts_fired

    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        if self._alert_engine is None:
            return [message]

        # Map source types to event topics for the alert engine
        source = message.source or message.data.get("source", "")
        topic = f"sensor.{source}" if source else "pipeline.message"

        try:
            alerts = self._alert_engine.evaluate_event(topic, message.data)
            if alerts:
                self._alerts_fired += len(alerts)
                message.data["alerts"] = [a.to_dict() for a in alerts]
        except Exception as exc:
            message.errors.append(f"AlertingStage: {exc}")

        return [message]


# ---------------------------------------------------------------------------
# Built-in stage: ReportingStage
# ---------------------------------------------------------------------------

class ReportingStage(PipelineStage):
    """Generates periodic situation reports from pipeline data.

    Accumulates messages and generates a report summary at configurable
    intervals. Individual messages pass through unchanged.

    Parameters
    ----------
    interval:
        Seconds between report generations (default 3600 = 1 hour).
    report_callback:
        Optional callback that receives the report dict when generated.
    """

    def __init__(
        self,
        interval: float = 3600,
        *,
        report_callback: Callable[[dict], None] | None = None,
        name: str = "",
    ) -> None:
        super().__init__(name=name or "ReportingStage")
        self._interval = interval
        self._report_callback = report_callback
        self._last_report_time: float = 0.0
        self._message_buffer: list[dict] = []
        self._reports_generated: int = 0
        self._lock = threading.Lock()

    @property
    def interval(self) -> float:
        return self._interval

    @property
    def reports_generated(self) -> int:
        return self._reports_generated

    @property
    def last_report(self) -> dict | None:
        """Return the most recent generated report, or None."""
        with self._lock:
            if not hasattr(self, "_last_report_data"):
                return None
            return self._last_report_data

    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        now = time.time()

        with self._lock:
            self._message_buffer.append(message.to_dict())

            if self._last_report_time == 0.0:
                self._last_report_time = now

            if (now - self._last_report_time) >= self._interval:
                self._generate_report(now)

        return [message]

    def force_report(self) -> dict:
        """Force generation of a report right now, regardless of interval."""
        with self._lock:
            return self._generate_report(time.time())

    def _generate_report(self, now: float) -> dict:
        """Generate a report from the buffered messages. Must hold self._lock."""
        # Count by source
        source_counts: dict[str, int] = {}
        alert_count = 0
        error_count = 0
        for msg in self._message_buffer:
            src = msg.get("source", "unknown")
            source_counts[src] = source_counts.get(src, 0) + 1
            if msg.get("data", {}).get("alerts"):
                alert_count += len(msg["data"]["alerts"])
            if msg.get("errors"):
                error_count += len(msg["errors"])

        report = {
            "report_id": uuid.uuid4().hex[:12],
            "generated_at": now,
            "period_start": self._last_report_time,
            "period_end": now,
            "period_seconds": now - self._last_report_time,
            "total_messages": len(self._message_buffer),
            "source_counts": source_counts,
            "alert_count": alert_count,
            "error_count": error_count,
        }

        self._last_report_data = report
        self._reports_generated += 1
        self._last_report_time = now
        self._message_buffer.clear()

        if self._report_callback:
            try:
                self._report_callback(report)
            except Exception as exc:
                logger.warning("Report callback error: %s", exc)

        logger.info(
            "Pipeline report generated: %d messages, %d alerts",
            report["total_messages"],
            report["alert_count"],
        )

        return report


# ---------------------------------------------------------------------------
# Pipeline — the chain executor
# ---------------------------------------------------------------------------

class Pipeline:
    """Configurable chain of processing stages.

    Messages flow through stages in order. Each stage can transform,
    filter, or fan-out messages. The pipeline tracks per-stage stats
    and supports lifecycle management.

    Parameters
    ----------
    stages:
        Ordered list of PipelineStage instances.
    config:
        Optional PipelineConfig for additional settings.
    on_complete:
        Optional callback invoked for each message that reaches the end.
    on_drop:
        Optional callback invoked when a message is dropped by a stage.
    on_error:
        Optional callback invoked when a stage error occurs.
    """

    def __init__(
        self,
        stages: list[PipelineStage],
        *,
        config: PipelineConfig | None = None,
        on_complete: Callable[[PipelineMessage], None] | None = None,
        on_drop: Callable[[PipelineMessage, str], None] | None = None,
        on_error: Callable[[PipelineMessage, str, str], None] | None = None,
    ) -> None:
        self._stages = list(stages)
        self._config = config or PipelineConfig()
        self._on_complete = on_complete
        self._on_drop = on_drop
        self._on_error = on_error
        self._state = PipelineState.CREATED
        self._lock = threading.Lock()

        # Stats
        self._total_pushed: int = 0
        self._total_completed: int = 0
        self._total_dropped: int = 0
        self._total_errors: int = 0
        self._start_time: float = 0.0

    @property
    def state(self) -> PipelineState:
        return self._state

    @property
    def stages(self) -> list[PipelineStage]:
        return list(self._stages)

    @property
    def stage_names(self) -> list[str]:
        return [s.name for s in self._stages]

    def start(self) -> None:
        """Initialize all stages and mark the pipeline as running."""
        if self._state == PipelineState.RUNNING:
            return

        for stage in self._stages:
            try:
                stage.setup()
            except Exception as exc:
                logger.error("Stage %s setup failed: %s", stage.name, exc)
                self._state = PipelineState.ERROR
                raise

        self._state = PipelineState.RUNNING
        self._start_time = time.time()
        logger.info(
            "Pipeline '%s' started with %d stages: %s",
            self._config.name,
            len(self._stages),
            " -> ".join(s.name for s in self._stages),
        )

    def stop(self) -> None:
        """Tear down all stages and mark the pipeline as stopped."""
        if self._state == PipelineState.STOPPED:
            return

        for stage in self._stages:
            try:
                stage.teardown()
            except Exception as exc:
                logger.warning("Stage %s teardown error: %s", stage.name, exc)

        self._state = PipelineState.STOPPED
        logger.info("Pipeline '%s' stopped", self._config.name)

    def pause(self) -> None:
        """Pause the pipeline. Messages pushed while paused are rejected."""
        if self._state == PipelineState.RUNNING:
            self._state = PipelineState.PAUSED

    def resume(self) -> None:
        """Resume a paused pipeline."""
        if self._state == PipelineState.PAUSED:
            self._state = PipelineState.RUNNING

    def push(self, data: dict) -> PipelineMessage | None:
        """Push raw data through the pipeline.

        Parameters
        ----------
        data:
            Raw sensor data dict. Must contain at least a ``source`` key
            or enough fields for the IngestStage to infer the source.

        Returns
        -------
        PipelineMessage | None:
            The final message after all stages, or None if dropped or
            if the pipeline is not running.
        """
        if self._state != PipelineState.RUNNING:
            return None

        message = PipelineMessage(
            data=dict(data),
            source=data.get("source", ""),
        )

        with self._lock:
            self._total_pushed += 1

        return self._execute(message)

    def push_message(self, message: PipelineMessage) -> PipelineMessage | None:
        """Push a pre-constructed PipelineMessage through the pipeline.

        Parameters
        ----------
        message:
            A PipelineMessage to process.

        Returns
        -------
        PipelineMessage | None:
            The final message after all stages, or None if dropped.
        """
        if self._state != PipelineState.RUNNING:
            return None

        with self._lock:
            self._total_pushed += 1

        return self._execute(message)

    def _execute(self, message: PipelineMessage) -> PipelineMessage | None:
        """Run a message through all stages sequentially."""
        messages = [message]

        for stage in self._stages:
            next_messages: list[PipelineMessage] = []
            for msg in messages:
                results = stage._run(msg)

                # Handle errors per policy
                for r in results:
                    if r.errors and self._config.error_policy == "drop":
                        r.dropped = True
                        with self._lock:
                            self._total_dropped += 1
                            self._total_errors += 1
                        if self._on_error:
                            self._on_error(r, stage.name, r.errors[-1])
                        if self._on_drop:
                            self._on_drop(r, stage.name)
                        continue
                    elif r.errors and self._config.error_policy == "halt":
                        self._state = PipelineState.ERROR
                        with self._lock:
                            self._total_errors += 1
                        if self._on_error:
                            self._on_error(r, stage.name, r.errors[-1])
                        return None
                    next_messages.append(r)

                if not results:
                    # Stage dropped the message
                    with self._lock:
                        self._total_dropped += 1
                    if self._on_drop:
                        self._on_drop(msg, stage.name)

            messages = next_messages
            if not messages:
                return None

        # Messages that made it through all stages
        for msg in messages:
            with self._lock:
                self._total_completed += 1
            if self._on_complete:
                self._on_complete(msg)

        # Return the first completed message (most common case: 1:1)
        return messages[0] if messages else None

    def push_batch(self, items: list[dict]) -> list[PipelineMessage]:
        """Push multiple data items through the pipeline.

        Parameters
        ----------
        items:
            List of raw data dicts.

        Returns
        -------
        list[PipelineMessage]:
            List of completed messages (dropped messages excluded).
        """
        results = []
        for item in items:
            result = self.push(item)
            if result is not None:
                results.append(result)
        return results

    def get_stats(self) -> dict[str, Any]:
        """Return pipeline-wide and per-stage statistics."""
        with self._lock:
            uptime = time.time() - self._start_time if self._start_time > 0 else 0
            return {
                "name": self._config.name,
                "state": self._state.value,
                "stage_count": len(self._stages),
                "stage_names": self.stage_names,
                "total_pushed": self._total_pushed,
                "total_completed": self._total_completed,
                "total_dropped": self._total_dropped,
                "total_errors": self._total_errors,
                "completion_rate": round(
                    self._total_completed / max(self._total_pushed, 1), 4
                ),
                "uptime_seconds": round(uptime, 1),
                "error_policy": self._config.error_policy,
                "stages": [s.stats for s in self._stages],
            }


# ---------------------------------------------------------------------------
# PipelineRunner — threaded runner with backpressure
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Runs a Pipeline in a background thread with a bounded input queue.

    Provides backpressure management: if the queue is full, ``submit()``
    blocks (or drops, depending on policy).

    Parameters
    ----------
    pipeline:
        The Pipeline to run.
    max_queue_size:
        Maximum number of pending messages. 0 = unbounded.
    drop_on_full:
        If True, silently drop messages when the queue is full instead
        of blocking.
    drain_timeout:
        Seconds to wait for queue drain on stop.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        *,
        max_queue_size: int = 10000,
        drop_on_full: bool = False,
        drain_timeout: float = 5.0,
    ) -> None:
        self._pipeline = pipeline
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._drop_on_full = drop_on_full
        self._drain_timeout = drain_timeout
        self._running = False
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

        # Stats
        self._submitted: int = 0
        self._dropped_backpressure: int = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def start(self) -> None:
        """Start the pipeline and the background consumer thread."""
        if self._running:
            return

        self._pipeline.start()
        self._running = True
        self._thread = threading.Thread(
            target=self._consumer_loop,
            name="pipeline_runner",
            daemon=True,
        )
        self._thread.start()
        logger.info("PipelineRunner started")

    def stop(self) -> None:
        """Stop the runner, drain remaining messages, and stop the pipeline."""
        if not self._running:
            return

        self._running = False

        # Signal consumer thread to exit
        try:
            self._queue.put(None, timeout=1.0)
        except queue.Full:
            pass

        if self._thread:
            self._thread.join(timeout=self._drain_timeout + 2)
            self._thread = None

        self._pipeline.stop()
        logger.info("PipelineRunner stopped")

    def submit(self, data: dict) -> bool:
        """Submit raw data to the pipeline queue.

        Parameters
        ----------
        data:
            Raw sensor data dict.

        Returns
        -------
        bool:
            True if accepted, False if dropped due to backpressure.
        """
        if not self._running:
            return False

        try:
            if self._drop_on_full:
                self._queue.put_nowait(data)
            else:
                self._queue.put(data, timeout=1.0)
            with self._lock:
                self._submitted += 1
            return True
        except queue.Full:
            with self._lock:
                self._dropped_backpressure += 1
            logger.debug("Pipeline backpressure: message dropped")
            return False

    def submit_batch(self, items: list[dict]) -> int:
        """Submit multiple items. Returns count of accepted items."""
        accepted = 0
        for item in items:
            if self.submit(item):
                accepted += 1
        return accepted

    def _consumer_loop(self) -> None:
        """Background thread that drains the queue through the pipeline."""
        while self._running:
            try:
                item = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if item is None:
                # Poison pill — drain remaining and exit
                while not self._queue.empty():
                    try:
                        remaining = self._queue.get_nowait()
                        if remaining is not None:
                            self._pipeline.push(remaining)
                    except queue.Empty:
                        break
                break

            self._pipeline.push(item)

    def get_stats(self) -> dict[str, Any]:
        """Return runner and pipeline statistics."""
        with self._lock:
            return {
                "running": self._running,
                "queue_size": self._queue.qsize(),
                "submitted": self._submitted,
                "dropped_backpressure": self._dropped_backpressure,
                "pipeline": self._pipeline.get_stats(),
            }


# ---------------------------------------------------------------------------
# Stage registry — factory for config-driven pipeline construction
# ---------------------------------------------------------------------------

_STAGE_REGISTRY: dict[str, type[PipelineStage]] = {}


def register_stage(name: str, cls: type[PipelineStage]) -> None:
    """Register a stage class for config-driven pipeline construction."""
    _STAGE_REGISTRY[name] = cls


def get_registered_stages() -> dict[str, type[PipelineStage]]:
    """Return a copy of the stage registry."""
    return dict(_STAGE_REGISTRY)


def build_pipeline_from_config(config: PipelineConfig, **kwargs: Any) -> Pipeline:
    """Construct a Pipeline from a PipelineConfig.

    Each stage dict in the config must have a ``type`` key matching
    a registered stage name. Remaining keys are passed as kwargs to
    the stage constructor.

    Parameters
    ----------
    config:
        The pipeline configuration.
    **kwargs:
        Additional keyword arguments passed to the Pipeline constructor
        (e.g., on_complete, on_drop, on_error).

    Returns
    -------
    Pipeline:
        A fully configured Pipeline ready to start.
    """
    stages: list[PipelineStage] = []
    for stage_def in config.stages:
        stage_type = stage_def.get("type", "")
        stage_cls = _STAGE_REGISTRY.get(stage_type)
        if stage_cls is None:
            raise ValueError(f"Unknown stage type: {stage_type!r}")
        stage_kwargs = {k: v for k, v in stage_def.items() if k != "type"}
        stages.append(stage_cls(**stage_kwargs))

    return Pipeline(stages, config=config, **kwargs)


# Register built-in stages
register_stage("ingest", IngestStage)
register_stage("tracking", TrackingStage)
register_stage("fusion", FusionStage)
register_stage("alerting", AlertingStage)
register_stage("reporting", ReportingStage)


# ---------------------------------------------------------------------------
# Convenience: default_pipeline()
# ---------------------------------------------------------------------------

def default_pipeline(**kwargs: Any) -> Pipeline:
    """Create a pipeline with all five built-in stages using default settings.

    Parameters
    ----------
    **kwargs:
        Passed through to Pipeline (on_complete, on_drop, on_error).
    """
    return Pipeline(
        [
            IngestStage(),
            TrackingStage(),
            FusionStage(),
            AlertingStage(),
            ReportingStage(),
        ],
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Exports
# ---------------------------------------------------------------------------

__all__ = [
    # Core
    "PipelineMessage",
    "PipelineState",
    "PipelineStage",
    "PipelineConfig",
    "Pipeline",
    "PipelineRunner",
    # Built-in stages
    "IngestStage",
    "TrackingStage",
    "FusionStage",
    "AlertingStage",
    "ReportingStage",
    "VALID_SOURCES",
    # Registry
    "register_stage",
    "get_registered_stages",
    "build_pipeline_from_config",
    # Convenience
    "default_pipeline",
]
