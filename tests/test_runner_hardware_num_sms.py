# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors
"""`runner.hardware_num_sms`: the SM count kernels are specialized for at prepare time."""

import sys

import pytest

from tirx_kernels import runner


@pytest.fixture
def cuda_not_initialized(monkeypatch):
    """Prepare-time conditions: no framework CUDA state in this process."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.delenv(runner.PREPARE_NUM_SMS_ENV, raising=False)


def test_environment_override_wins(monkeypatch, cuda_not_initialized):
    monkeypatch.setenv(runner.PREPARE_NUM_SMS_ENV, "100")
    monkeypatch.setattr(runner, "_probed_num_sms", lambda _: 132)
    assert runner.hardware_num_sms() == 100
    monkeypatch.setenv(runner.PREPARE_NUM_SMS_ENV, "0")
    with pytest.raises(ValueError, match="must be positive"):
        runner.hardware_num_sms()


def test_visible_device_is_queried(monkeypatch, cuda_not_initialized):
    monkeypatch.setattr(runner, "_probed_num_sms", lambda _: 132)
    assert runner.hardware_num_sms() == 132
    assert runner.hardware_num_sms(216) == 132  # a present device beats a caller default


def test_explicit_default_only_without_a_device(monkeypatch, cuda_not_initialized):
    monkeypatch.setattr(runner, "_probed_num_sms", lambda _: None)
    assert runner.hardware_num_sms(216) == 216


def test_no_device_and_no_default_fails_loudly(monkeypatch, cuda_not_initialized):
    monkeypatch.setattr(runner, "_probed_num_sms", lambda _: None)
    with pytest.raises(RuntimeError, match=runner.PREPARE_NUM_SMS_ENV):
        runner.hardware_num_sms()


def test_probe_reports_none_without_a_device(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    runner._probed_num_sms.cache_clear()
    try:
        assert runner._probed_num_sms("") is None
    finally:
        runner._probed_num_sms.cache_clear()


def test_probe_matches_the_device_and_leaves_cuda_uninitialized():
    contexts_before = runner._current_process_cuda_uuids()
    runner._probed_num_sms.cache_clear()
    try:
        probed = runner._probed_num_sms("probe-test")
    finally:
        runner._probed_num_sms.cache_clear()
    if probed is None:
        pytest.skip("no CUDA device visible")
    # The query ran in a child process: this one gained no CUDA context.
    assert runner._current_process_cuda_uuids() == contexts_before
    torch = pytest.importorskip("torch")
    assert probed == torch.cuda.get_device_properties(0).multi_processor_count
