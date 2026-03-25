# Created by Matthew Valancy
# Copyright 2026 Valpatel Software LLC
# Licensed under AGPL-3.0 — see LICENSE for details.
"""Tests for tritium_lib.models.benchmark — benchmark result models."""

from tritium_lib.models.benchmark import BenchmarkResult, BenchmarkSuite, BenchmarkUnit


class TestBenchmarkUnit:
    def test_unit_values(self):
        assert BenchmarkUnit.MILLISECONDS.value == "ms"
        assert BenchmarkUnit.SECONDS.value == "s"
        assert BenchmarkUnit.OPS_PER_SEC.value == "ops/s"
        assert BenchmarkUnit.MEGABYTES.value == "MB"
        assert BenchmarkUnit.PERCENT.value == "%"
        assert BenchmarkUnit.COUNT.value == "count"


class TestBenchmarkResult:
    def test_creation(self):
        r = BenchmarkResult(
            test_name="latency_test",
            metric="p99",
            value=15.5,
            unit=BenchmarkUnit.MILLISECONDS,
            threshold=20.0,
            passed=True,
        )
        assert r.test_name == "latency_test"
        assert r.metric == "p99"
        assert r.value == 15.5
        assert r.passed is True

    def test_summary_pass(self):
        r = BenchmarkResult(
            test_name="test1", metric="latency",
            value=10.0, unit=BenchmarkUnit.MILLISECONDS,
            threshold=20.0, passed=True,
        )
        s = r.summary()
        assert "[PASS]" in s
        assert "test1" in s
        assert "latency" in s

    def test_summary_fail(self):
        r = BenchmarkResult(
            test_name="test1", metric="latency",
            value=30.0, unit=BenchmarkUnit.MILLISECONDS,
            threshold=20.0, passed=False,
        )
        s = r.summary()
        assert "[FAIL]" in s

    def test_timestamp_auto_set(self):
        r = BenchmarkResult(
            test_name="t", metric="m",
            value=1.0, unit=BenchmarkUnit.COUNT,
            threshold=10.0, passed=True,
        )
        assert r.timestamp is not None

    def test_metadata_default(self):
        r = BenchmarkResult(
            test_name="t", metric="m",
            value=1.0, unit=BenchmarkUnit.COUNT,
            threshold=10.0, passed=True,
        )
        assert r.metadata == {}


class TestBenchmarkSuite:
    def test_empty_suite(self):
        s = BenchmarkSuite(suite_name="empty")
        assert s.all_passed is True
        assert s.pass_count == 0
        assert s.fail_count == 0

    def test_add_passing_result(self):
        s = BenchmarkSuite(suite_name="test")
        r = s.add("test1", "latency", 10.0, BenchmarkUnit.MILLISECONDS, 20.0)
        assert r.passed is True
        assert s.pass_count == 1
        assert s.fail_count == 0
        assert s.all_passed is True

    def test_add_failing_result(self):
        s = BenchmarkSuite(suite_name="test")
        r = s.add("test1", "latency", 30.0, BenchmarkUnit.MILLISECONDS, 20.0)
        assert r.passed is False
        assert s.pass_count == 0
        assert s.fail_count == 1
        assert s.all_passed is False

    def test_higher_is_better(self):
        s = BenchmarkSuite(suite_name="throughput")
        r = s.add("ops_test", "throughput", 1000.0,
                   BenchmarkUnit.OPS_PER_SEC, 500.0, higher_is_better=True)
        assert r.passed is True

    def test_higher_is_better_fail(self):
        s = BenchmarkSuite(suite_name="throughput")
        r = s.add("ops_test", "throughput", 200.0,
                   BenchmarkUnit.OPS_PER_SEC, 500.0, higher_is_better=True)
        assert r.passed is False

    def test_mixed_results(self):
        s = BenchmarkSuite(suite_name="mixed")
        s.add("t1", "m1", 5.0, BenchmarkUnit.MILLISECONDS, 10.0)
        s.add("t2", "m2", 15.0, BenchmarkUnit.MILLISECONDS, 10.0)
        assert s.pass_count == 1
        assert s.fail_count == 1
        assert s.all_passed is False

    def test_report(self):
        s = BenchmarkSuite(suite_name="report_test")
        s.add("t1", "latency", 5.0, BenchmarkUnit.MILLISECONDS, 10.0)
        s.add("t2", "memory", 150.0, BenchmarkUnit.MEGABYTES, 100.0)
        report = s.report()
        assert "report_test" in report
        assert "1 passed" in report
        assert "1 failed" in report

    def test_to_dict(self):
        s = BenchmarkSuite(suite_name="serialization_test")
        s.add("t1", "m", 5.0, BenchmarkUnit.COUNT, 10.0)
        d = s.to_dict()
        assert d["suite_name"] == "serialization_test"
        assert d["all_passed"] is True
        assert d["pass_count"] == 1
        assert d["fail_count"] == 0
        assert len(d["results"]) == 1

    def test_add_with_metadata(self):
        s = BenchmarkSuite(suite_name="meta")
        r = s.add("t1", "m", 5.0, BenchmarkUnit.COUNT, 10.0,
                   metadata={"hardware": "test-box"})
        assert r.metadata["hardware"] == "test-box"

    def test_environment_field(self):
        s = BenchmarkSuite(
            suite_name="env_test",
            environment={"cpu": "x86", "os": "linux"},
        )
        d = s.to_dict()
        assert d["environment"]["cpu"] == "x86"
