# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.pipeline — data pipeline orchestrator."""

import time
import threading
import pytest

from tritium_lib.pipeline import (
    Pipeline,
    PipelineConfig,
    PipelineMessage,
    PipelineRunner,
    PipelineStage,
    PipelineState,
    IngestStage,
    TrackingStage,
    FusionStage,
    AlertingStage,
    ReportingStage,
    VALID_SOURCES,
    register_stage,
    get_registered_stages,
    build_pipeline_from_config,
    default_pipeline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class PassthroughStage(PipelineStage):
    """Test stage that passes messages through unchanged."""

    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        return [message]


class DropStage(PipelineStage):
    """Test stage that drops all messages."""

    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        return []


class TransformStage(PipelineStage):
    """Test stage that adds a field to message data."""

    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        message.data["transformed"] = True
        return [message]


class FanOutStage(PipelineStage):
    """Test stage that duplicates each message."""

    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        clone = message.clone()
        clone.data["copy"] = True
        return [message, clone]


class ErrorStage(PipelineStage):
    """Test stage that always raises an exception."""

    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        raise ValueError("intentional test error")


class CountingStage(PipelineStage):
    """Test stage that counts messages."""

    def __init__(self, name: str = "CountingStage") -> None:
        super().__init__(name=name)
        self.count = 0

    def process(self, message: PipelineMessage) -> list[PipelineMessage]:
        self.count += 1
        return [message]


def _ble_data(mac: str = "AA:BB:CC:DD:EE:FF", rssi: int = -55) -> dict:
    return {"source": "ble", "mac": mac, "rssi": rssi}


def _wifi_data(mac: str = "11:22:33:44:55:66", ssid: str = "TestNet") -> dict:
    return {"source": "wifi", "mac": mac, "ssid": ssid, "rssi": -70}


def _camera_data(cls: str = "person", conf: float = 0.9) -> dict:
    return {
        "source": "camera",
        "class_name": cls,
        "confidence": conf,
        "center_x": 10.0,
        "center_y": 5.0,
    }


# ---------------------------------------------------------------------------
# PipelineMessage tests
# ---------------------------------------------------------------------------

class TestPipelineMessage:

    def test_create_message(self):
        msg = PipelineMessage(data={"foo": "bar"}, source="test")
        assert msg.source == "test"
        assert msg.data["foo"] == "bar"
        assert len(msg.message_id) == 12
        assert msg.stage_trace == []
        assert msg.errors == []
        assert msg.dropped is False

    def test_clone_message(self):
        msg = PipelineMessage(data={"key": "value"}, source="ble")
        msg.stage_trace.append("stage1")
        clone = msg.clone()
        assert clone.message_id != msg.message_id
        assert clone.data == msg.data
        assert clone.source == msg.source
        assert clone.stage_trace == msg.stage_trace
        # Ensure independence
        clone.data["new_key"] = "new_value"
        assert "new_key" not in msg.data

    def test_to_dict(self):
        msg = PipelineMessage(data={"x": 1}, source="wifi")
        d = msg.to_dict()
        assert d["source"] == "wifi"
        assert d["data"]["x"] == 1
        assert "message_id" in d
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# PipelineConfig tests
# ---------------------------------------------------------------------------

class TestPipelineConfig:

    def test_default_config(self):
        cfg = PipelineConfig()
        assert cfg.name == "default"
        assert cfg.max_queue_size == 10000
        assert cfg.error_policy == "propagate"

    def test_from_dict(self):
        d = {
            "name": "my_pipeline",
            "stages": [{"type": "ingest"}, {"type": "tracking"}],
            "max_queue_size": 500,
            "error_policy": "drop",
        }
        cfg = PipelineConfig.from_dict(d)
        assert cfg.name == "my_pipeline"
        assert len(cfg.stages) == 2
        assert cfg.max_queue_size == 500
        assert cfg.error_policy == "drop"

    def test_to_dict(self):
        cfg = PipelineConfig(name="test", max_queue_size=100)
        d = cfg.to_dict()
        assert d["name"] == "test"
        assert d["max_queue_size"] == 100


# ---------------------------------------------------------------------------
# PipelineStage tests
# ---------------------------------------------------------------------------

class TestPipelineStage:

    def test_passthrough_stage(self):
        stage = PassthroughStage(name="pass")
        msg = PipelineMessage(data={"a": 1})
        results = stage._run(msg)
        assert len(results) == 1
        assert results[0].data["a"] == 1
        assert "pass" in results[0].stage_trace

    def test_drop_stage(self):
        stage = DropStage(name="drop")
        msg = PipelineMessage(data={"a": 1})
        results = stage._run(msg)
        assert len(results) == 0
        assert stage.stats["dropped"] == 1

    def test_transform_stage(self):
        stage = TransformStage(name="xform")
        msg = PipelineMessage(data={"b": 2})
        results = stage._run(msg)
        assert len(results) == 1
        assert results[0].data["transformed"] is True
        assert results[0].data["b"] == 2

    def test_fanout_stage(self):
        stage = FanOutStage(name="fanout")
        msg = PipelineMessage(data={"c": 3})
        results = stage._run(msg)
        assert len(results) == 2
        # One original, one copy
        copies = [r for r in results if r.data.get("copy")]
        assert len(copies) == 1

    def test_error_stage_records_error(self):
        stage = ErrorStage(name="err")
        msg = PipelineMessage(data={})
        results = stage._run(msg)
        assert len(results) == 1
        assert len(results[0].errors) == 1
        assert "intentional test error" in results[0].errors[0]
        assert stage.stats["errors"] == 1

    def test_stage_stats(self):
        stage = CountingStage(name="counter")
        for i in range(5):
            stage._run(PipelineMessage(data={"i": i}))
        stats = stage.stats
        assert stats["processed"] == 5
        assert stats["errors"] == 0
        assert stats["name"] == "counter"

    def test_stage_default_name(self):
        stage = PassthroughStage()
        assert stage.name == "PassthroughStage"


# ---------------------------------------------------------------------------
# IngestStage tests
# ---------------------------------------------------------------------------

class TestIngestStage:

    def test_accept_ble(self):
        stage = IngestStage()
        msg = PipelineMessage(data=_ble_data())
        results = stage._run(msg)
        assert len(results) == 1
        assert results[0].source == "ble"
        assert "timestamp" in results[0].data

    def test_accept_wifi(self):
        stage = IngestStage()
        msg = PipelineMessage(data=_wifi_data())
        results = stage._run(msg)
        assert len(results) == 1
        assert results[0].source == "wifi"

    def test_accept_camera(self):
        stage = IngestStage()
        msg = PipelineMessage(data=_camera_data())
        results = stage._run(msg)
        assert len(results) == 1
        assert results[0].source == "camera"

    def test_reject_unknown_source(self):
        stage = IngestStage(sources=["ble"])
        msg = PipelineMessage(data={"source": "unknown_sensor"})
        results = stage._run(msg)
        assert len(results) == 0

    def test_filter_by_source_list(self):
        stage = IngestStage(sources=["ble", "camera"])
        # BLE should pass
        msg_ble = PipelineMessage(data=_ble_data())
        assert len(stage._run(msg_ble)) == 1
        # WiFi should be filtered
        msg_wifi = PipelineMessage(data=_wifi_data())
        assert len(stage._run(msg_wifi)) == 0

    def test_infer_source_from_data(self):
        stage = IngestStage()
        # MAC + RSSI => BLE
        msg = PipelineMessage(data={"mac": "AA:BB:CC:DD:EE:FF", "rssi": -50})
        results = stage._run(msg)
        assert len(results) == 1
        assert results[0].source == "ble"

    def test_infer_camera_from_class_name(self):
        stage = IngestStage()
        msg = PipelineMessage(data={"class_name": "car", "confidence": 0.8})
        results = stage._run(msg)
        assert len(results) == 1
        assert results[0].source == "camera"

    def test_sources_property(self):
        stage = IngestStage(sources=["ble", "wifi"])
        assert stage.sources == {"ble", "wifi"}

    def test_valid_sources_constant(self):
        assert "ble" in VALID_SOURCES
        assert "wifi" in VALID_SOURCES
        assert "camera" in VALID_SOURCES
        assert "acoustic" in VALID_SOURCES
        assert "mesh" in VALID_SOURCES
        assert "adsb" in VALID_SOURCES
        assert "rf_motion" in VALID_SOURCES


# ---------------------------------------------------------------------------
# Pipeline chain tests
# ---------------------------------------------------------------------------

class TestPipeline:

    def test_create_pipeline(self):
        p = Pipeline([PassthroughStage()])
        assert p.state == PipelineState.CREATED
        assert len(p.stages) == 1

    def test_start_stop(self):
        p = Pipeline([PassthroughStage()])
        p.start()
        assert p.state == PipelineState.RUNNING
        p.stop()
        assert p.state == PipelineState.STOPPED

    def test_push_data_through(self):
        p = Pipeline([PassthroughStage(), TransformStage()])
        p.start()
        result = p.push({"value": 42})
        assert result is not None
        assert result.data["value"] == 42
        assert result.data["transformed"] is True
        p.stop()

    def test_push_returns_none_when_not_running(self):
        p = Pipeline([PassthroughStage()])
        result = p.push({"x": 1})
        assert result is None

    def test_stage_trace(self):
        p = Pipeline([
            PassthroughStage(name="A"),
            PassthroughStage(name="B"),
            PassthroughStage(name="C"),
        ])
        p.start()
        result = p.push({"data": True})
        assert result is not None
        assert result.stage_trace == ["A", "B", "C"]
        p.stop()

    def test_drop_stage_in_chain(self):
        completed = []
        p = Pipeline(
            [PassthroughStage(), DropStage(), PassthroughStage()],
            on_complete=lambda m: completed.append(m),
        )
        p.start()
        result = p.push({"x": 1})
        assert result is None
        assert len(completed) == 0
        p.stop()

    def test_on_complete_callback(self):
        completed = []
        p = Pipeline(
            [PassthroughStage()],
            on_complete=lambda m: completed.append(m),
        )
        p.start()
        p.push({"test": True})
        assert len(completed) == 1
        assert completed[0].data["test"] is True
        p.stop()

    def test_on_drop_callback(self):
        dropped = []
        p = Pipeline(
            [DropStage()],
            on_drop=lambda m, s: dropped.append((m, s)),
        )
        p.start()
        p.push({"x": 1})
        assert len(dropped) == 1
        assert dropped[0][1] == "DropStage"
        p.stop()

    def test_error_policy_propagate(self):
        p = Pipeline(
            [ErrorStage()],
            config=PipelineConfig(error_policy="propagate"),
        )
        p.start()
        result = p.push({"x": 1})
        # Error is propagated: message passes through with error attached
        assert result is not None
        assert len(result.errors) == 1
        p.stop()

    def test_error_policy_drop(self):
        p = Pipeline(
            [ErrorStage()],
            config=PipelineConfig(error_policy="drop"),
        )
        p.start()
        result = p.push({"x": 1})
        assert result is None
        stats = p.get_stats()
        assert stats["total_dropped"] == 1
        p.stop()

    def test_error_policy_halt(self):
        p = Pipeline(
            [ErrorStage()],
            config=PipelineConfig(error_policy="halt"),
        )
        p.start()
        result = p.push({"x": 1})
        assert result is None
        assert p.state == PipelineState.ERROR
        p.stop()

    def test_pause_and_resume(self):
        p = Pipeline([PassthroughStage()])
        p.start()
        assert p.state == PipelineState.RUNNING
        p.pause()
        assert p.state == PipelineState.PAUSED
        result = p.push({"x": 1})
        assert result is None  # Rejected while paused
        p.resume()
        assert p.state == PipelineState.RUNNING
        result = p.push({"x": 2})
        assert result is not None
        p.stop()

    def test_push_batch(self):
        counter = CountingStage()
        p = Pipeline([counter])
        p.start()
        results = p.push_batch([{"i": 0}, {"i": 1}, {"i": 2}])
        assert len(results) == 3
        assert counter.count == 3
        p.stop()

    def test_push_message(self):
        p = Pipeline([PassthroughStage()])
        p.start()
        msg = PipelineMessage(data={"custom": True}, source="test")
        result = p.push_message(msg)
        assert result is not None
        assert result.data["custom"] is True
        p.stop()

    def test_pipeline_stats(self):
        p = Pipeline(
            [IngestStage(), PassthroughStage(name="proc")],
            config=PipelineConfig(name="test_pipeline"),
        )
        p.start()
        p.push(_ble_data())
        p.push(_camera_data())
        stats = p.get_stats()
        assert stats["name"] == "test_pipeline"
        assert stats["state"] == "running"
        assert stats["total_pushed"] == 2
        assert stats["total_completed"] == 2
        assert stats["stage_count"] == 2
        assert len(stats["stages"]) == 2
        p.stop()

    def test_stage_names(self):
        p = Pipeline([
            IngestStage(),
            TrackingStage(),
            FusionStage(),
        ])
        assert p.stage_names == ["IngestStage", "TrackingStage", "FusionStage"]

    def test_fanout_through_pipeline(self):
        completed = []
        p = Pipeline(
            [FanOutStage(), CountingStage()],
            on_complete=lambda m: completed.append(m),
        )
        p.start()
        p.push({"x": 1})
        # FanOut produces 2, Counter sees 2
        assert len(completed) == 2
        p.stop()


# ---------------------------------------------------------------------------
# ReportingStage tests
# ---------------------------------------------------------------------------

class TestReportingStage:

    def test_report_generation(self):
        reports = []
        stage = ReportingStage(
            interval=0,  # immediate
            report_callback=lambda r: reports.append(r),
        )
        # First message sets the timer
        msg1 = PipelineMessage(data={"source": "ble"}, source="ble")
        stage._run(msg1)
        # With interval=0, the next message should trigger a report
        msg2 = PipelineMessage(data={"source": "wifi"}, source="wifi")
        stage._run(msg2)
        assert len(reports) >= 1
        report = reports[0]
        assert "report_id" in report
        assert "total_messages" in report
        assert "source_counts" in report

    def test_force_report(self):
        stage = ReportingStage(interval=9999)  # Won't trigger naturally
        msg = PipelineMessage(data={"source": "ble"}, source="ble")
        stage._run(msg)
        report = stage.force_report()
        assert report["total_messages"] == 1
        assert stage.reports_generated == 1

    def test_report_source_counts(self):
        stage = ReportingStage(interval=9999)
        for _ in range(3):
            stage._run(PipelineMessage(data={"source": "ble"}, source="ble"))
        for _ in range(2):
            stage._run(PipelineMessage(data={"source": "wifi"}, source="wifi"))
        report = stage.force_report()
        assert report["source_counts"]["ble"] == 3
        assert report["source_counts"]["wifi"] == 2
        assert report["total_messages"] == 5

    def test_last_report_property(self):
        stage = ReportingStage(interval=9999)
        assert stage.last_report is None
        stage._run(PipelineMessage(data={"source": "ble"}, source="ble"))
        stage.force_report()
        assert stage.last_report is not None


# ---------------------------------------------------------------------------
# TrackingStage tests
# ---------------------------------------------------------------------------

class TestTrackingStage:

    def test_tracking_stage_setup(self):
        stage = TrackingStage()
        stage.setup()
        assert stage.tracker is not None

    def test_tracking_stage_with_ble(self):
        from tritium_lib.tracking import TargetTracker
        tracker = TargetTracker()
        stage = TrackingStage(tracker=tracker)
        msg = PipelineMessage(
            data=_ble_data(),
            source="ble",
        )
        results = stage._run(msg)
        assert len(results) == 1
        assert results[0].data.get("tracked") is True


# ---------------------------------------------------------------------------
# FusionStage tests
# ---------------------------------------------------------------------------

class TestFusionStage:

    def test_fusion_stage_setup(self):
        stage = FusionStage()
        stage.setup()
        assert stage.engine is not None

    def test_fusion_stage_with_ble(self):
        from tritium_lib.fusion import FusionEngine
        engine = FusionEngine()
        stage = FusionStage(engine=engine)
        msg = PipelineMessage(
            data=_ble_data(),
            source="ble",
        )
        results = stage._run(msg)
        assert len(results) == 1
        assert "fused_target_id" in results[0].data


# ---------------------------------------------------------------------------
# AlertingStage tests
# ---------------------------------------------------------------------------

class TestAlertingStage:

    def test_alerting_stage_setup(self):
        stage = AlertingStage()
        stage.setup()
        assert stage.alert_engine is not None

    def test_alerting_stage_processes_message(self):
        stage = AlertingStage()
        stage.setup()
        msg = PipelineMessage(data=_ble_data(), source="ble")
        results = stage._run(msg)
        assert len(results) == 1
        # May or may not fire alerts depending on rules, but should not error
        assert stage.stats["errors"] == 0


# ---------------------------------------------------------------------------
# PipelineRunner tests
# ---------------------------------------------------------------------------

class TestPipelineRunner:

    def test_runner_start_stop(self):
        p = Pipeline([PassthroughStage()])
        runner = PipelineRunner(p)
        runner.start()
        assert runner.is_running is True
        runner.stop()
        assert runner.is_running is False

    def test_runner_submit(self):
        counter = CountingStage()
        p = Pipeline([counter])
        runner = PipelineRunner(p, max_queue_size=100)
        runner.start()
        for i in range(10):
            runner.submit({"i": i})
        # Give the consumer thread time to process
        time.sleep(0.5)
        runner.stop()
        assert counter.count == 10

    def test_runner_submit_batch(self):
        counter = CountingStage()
        p = Pipeline([counter])
        runner = PipelineRunner(p, max_queue_size=100)
        runner.start()
        accepted = runner.submit_batch([{"i": j} for j in range(5)])
        assert accepted == 5
        time.sleep(0.5)
        runner.stop()
        assert counter.count == 5

    def test_runner_backpressure_drop(self):
        # Tiny queue + slow stage
        p = Pipeline([PassthroughStage()])
        runner = PipelineRunner(p, max_queue_size=2, drop_on_full=True)
        runner.start()
        # Submit more than queue can hold without processing
        # Some should be dropped due to backpressure
        results = []
        for i in range(20):
            results.append(runner.submit({"i": i}))
        time.sleep(0.5)
        runner.stop()
        stats = runner.get_stats()
        assert stats["submitted"] + stats["dropped_backpressure"] == sum(1 for r in results)

    def test_runner_stats(self):
        p = Pipeline([PassthroughStage()])
        runner = PipelineRunner(p)
        runner.start()
        runner.submit({"x": 1})
        time.sleep(0.3)
        stats = runner.get_stats()
        assert "running" in stats
        assert "pipeline" in stats
        assert stats["submitted"] == 1
        runner.stop()

    def test_runner_submit_when_stopped(self):
        p = Pipeline([PassthroughStage()])
        runner = PipelineRunner(p)
        assert runner.submit({"x": 1}) is False


# ---------------------------------------------------------------------------
# Stage registry & config-driven construction tests
# ---------------------------------------------------------------------------

class TestStageRegistry:

    def test_builtin_stages_registered(self):
        stages = get_registered_stages()
        assert "ingest" in stages
        assert "tracking" in stages
        assert "fusion" in stages
        assert "alerting" in stages
        assert "reporting" in stages

    def test_register_custom_stage(self):
        register_stage("passthrough", PassthroughStage)
        stages = get_registered_stages()
        assert "passthrough" in stages

    def test_build_from_config(self):
        config = PipelineConfig(
            name="test",
            stages=[
                {"type": "ingest", "sources": ["ble"]},
                {"type": "reporting", "interval": 300},
            ],
        )
        p = build_pipeline_from_config(config)
        assert len(p.stages) == 2
        assert p.stages[0].name == "IngestStage"
        assert p.stages[1].name == "ReportingStage"

    def test_build_from_config_unknown_type(self):
        config = PipelineConfig(
            name="bad",
            stages=[{"type": "nonexistent"}],
        )
        with pytest.raises(ValueError, match="Unknown stage type"):
            build_pipeline_from_config(config)


# ---------------------------------------------------------------------------
# default_pipeline tests
# ---------------------------------------------------------------------------

class TestDefaultPipeline:

    def test_default_pipeline_has_five_stages(self):
        p = default_pipeline()
        assert len(p.stages) == 5
        names = p.stage_names
        assert names == [
            "IngestStage",
            "TrackingStage",
            "FusionStage",
            "AlertingStage",
            "ReportingStage",
        ]

    def test_default_pipeline_runs(self):
        p = default_pipeline()
        p.start()
        result = p.push(_ble_data())
        assert result is not None
        assert result.source == "ble"
        p.stop()


# ---------------------------------------------------------------------------
# Integration: full pipeline flow
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_full_pipeline_ble_flow(self):
        """End-to-end: BLE data through all five stages."""
        completed = []
        p = Pipeline(
            [
                IngestStage(),
                TrackingStage(),
                FusionStage(),
                AlertingStage(),
                ReportingStage(interval=9999),
            ],
            on_complete=lambda m: completed.append(m),
        )
        p.start()

        result = p.push(_ble_data())
        assert result is not None
        assert result.source == "ble"
        assert "fused_target_id" in result.data
        assert len(completed) == 1

        stats = p.get_stats()
        assert stats["total_completed"] == 1
        assert stats["total_dropped"] == 0
        p.stop()

    def test_full_pipeline_multi_source(self):
        """Multiple sensor types through the pipeline."""
        p = Pipeline([
            IngestStage(),
            TrackingStage(),
            FusionStage(),
            AlertingStage(),
            ReportingStage(interval=9999),
        ])
        p.start()

        results = p.push_batch([
            _ble_data(),
            _camera_data(),
            _wifi_data(),
        ])
        assert len(results) == 3

        stats = p.get_stats()
        assert stats["total_pushed"] == 3
        assert stats["total_completed"] == 3
        p.stop()

    def test_pipeline_with_runner_integration(self):
        """Pipeline through the threaded runner."""
        counter = CountingStage()
        p = Pipeline([IngestStage(), counter])
        runner = PipelineRunner(p, max_queue_size=100)
        runner.start()

        for _ in range(5):
            runner.submit(_ble_data())
        for _ in range(3):
            runner.submit(_camera_data())

        time.sleep(1.0)
        runner.stop()

        assert counter.count == 8
        stats = runner.get_stats()
        assert stats["submitted"] == 8
