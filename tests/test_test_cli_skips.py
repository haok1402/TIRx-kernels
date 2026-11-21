"""`python -m tirx_kernels.test` must not look green when it verified nothing."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest import SkipTest

import pytest

from tirx_kernels.test import __main__ as cli

REASON = "nvidia-cutlass-dsl: requires ==4.8.0.dev0, installed version is 4.7.0"


@pytest.fixture
def fake_registry(monkeypatch):
    def install(outcomes):
        module = SimpleNamespace(CONFIGS=[{"label": label} for label in outcomes])

        def run(name, config, **kwargs):
            outcome = outcomes[config["label"]]
            if outcome == "skip":
                raise SkipTest(REASON)
            if outcome == "fail":
                raise AssertionError("mismatch")

        monkeypatch.setattr(cli, "discover_kernels", lambda cuda_arch=None: {"k": module})
        monkeypatch.setattr(cli, "run_kernel_test", run)

    return install


def _main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["tirx_kernels.test", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    return exit_info.value.code


def test_all_skipped_fails_and_says_why(fake_registry, monkeypatch, capsys):
    fake_registry({"a": "skip", "b": "skip"})
    assert _main(monkeypatch, "--kernel", "k") == 3
    out, err = capsys.readouterr()
    assert f"skipped 2x: {REASON}" in out
    assert "nothing was verified" in err


def test_all_skipped_can_be_accepted_explicitly(fake_registry, monkeypatch):
    fake_registry({"a": "skip"})
    assert _main(monkeypatch, "--allow-all-skipped") == 0


def test_json_reports_skip_reasons(fake_registry, monkeypatch, capsys):
    fake_registry({"a": "skip", "b": "pass"})
    assert _main(monkeypatch, "--json") == 0  # something passed: partial skips stay allowed
    payload = json.loads(capsys.readouterr().out)
    assert payload["skip_reasons"] == [REASON]
    assert payload["all_skipped"] is False


def test_failures_still_exit_1(fake_registry, monkeypatch):
    fake_registry({"a": "skip", "b": "fail"})
    assert _main(monkeypatch) == 1
