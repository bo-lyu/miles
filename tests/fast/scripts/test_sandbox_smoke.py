"""Offline checks of the sandbox-smoke harness: the registry's shape and the
axis wiring. The real run needs a sandbox credential and is invoked manually
(see scripts/sandbox_smoke/README.md)."""

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_PY = REPO_ROOT / "scripts" / "sandbox_smoke" / "run.py"
OPENENV_DIR = REPO_ROOT / "examples" / "experimental" / "openenv"


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sandbox_smoke_run", RUN_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_connectors_are_callables(smoke):
    assert "harbor" in smoke.CONNECTORS
    for name, runner in smoke.CONNECTORS.items():
        assert callable(runner), name


def test_golden_run_wires_all_axes_through(smoke, monkeypatch, tmp_path):
    """connector/backend/agent/task all reach the runner; the golden default needs no endpoint."""
    seen = {}

    async def fake_runner(tasks_dir, task, *, backend, agent, base_url):
        seen.update(tasks_dir=tasks_dir, task=task, backend=backend, agent=agent, base_url=base_url)
        return {"reward": 1.0, "exit_status": "Submitted"}

    monkeypatch.setitem(smoke.CONNECTORS, "harbor", fake_runner)
    monkeypatch.setenv("TB2_TASKS_DIR", str(tmp_path))  # the env var wins over the cached clone
    monkeypatch.setattr("sys.argv", ["run.py", "--connector", "harbor", "--backend", "daytona"])
    assert smoke.main() == 0
    # the benchmark preset filled in the task; the golden default reaches the connector untranslated
    assert seen["backend"] == "daytona" and seen["task"] == "fix-git" and seen["agent"] == smoke.GOLDEN
    assert seen["tasks_dir"] == tmp_path


def test_a_real_harness_requires_a_model_endpoint(smoke, monkeypatch, capsys):
    monkeypatch.setattr(
        "sys.argv", ["run.py", "--connector", "harbor", "--backend", "e2b", "--agent", "mini-swe-agent"]
    )
    with pytest.raises(SystemExit) as excinfo:
        smoke.main()
    assert excinfo.value.code == 2
    assert "--base-url" in capsys.readouterr().err


# --- openenv connector --------------------------------------------------------
#
# _run_openenv reaches openenv_sandbox_common by inserting OPENENV_DIR onto
# sys.path and importing it by bare name (same as _run_harbor does for
# harbor_agent_function), so the module object it resolves to is the one these
# tests import here too and can monkeypatch.
if str(OPENENV_DIR) not in sys.path:
    sys.path.insert(0, str(OPENENV_DIR))
import openenv_sandbox_common as sandbox_common  # noqa: E402


def test_openenv_connector_only_runs_golden_for_now(smoke, tmp_path):
    """A harness agent needs #2802's exit_status vocabulary in openenv_agent_function
    first (see the docstring); until then this must fail loudly, not run and
    misreport the driver's PASS check."""
    with pytest.raises(NotImplementedError, match="golden"):
        asyncio.run(
            smoke._run_openenv(tmp_path, "fix-git", backend="e2b", agent="mini-swe-agent", base_url="http://x")
        )


def test_openenv_connector_maps_a_canonical_verdict_to_submitted(smoke, monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox_common, "load_backend", lambda name: object())

    async def fake_golden(backend, tasks_dir, task_id, **kwargs):
        assert tasks_dir == tmp_path and task_id == "fix-git"
        return {"reward": 1.0, "solve_exit": "0"}

    monkeypatch.setattr(sandbox_common, "run_golden_episode", fake_golden)

    result = asyncio.run(smoke._run_openenv(tmp_path, "fix-git", backend="e2b", agent=smoke.GOLDEN, base_url=""))
    assert result == {
        "reward": 1.0,
        "exit_status": "Submitted",
        "eval_report": {},
        "agent_metrics": {"solve_exit": "0"},
    }


def test_openenv_connector_maps_no_verdict_to_agent_error_not_a_false_zero(smoke, monkeypatch, tmp_path):
    """A missing verdict (server-side scoring failure, non-canonical harness) is
    the policy's -- nothing the golden solution did -- so it scores 0 with a
    named cause rather than discarding or silently passing."""
    monkeypatch.setattr(sandbox_common, "load_backend", lambda name: object())

    async def no_verdict(backend, tasks_dir, task_id, **kwargs):
        return {"reward": None, "error": "no canonical verdict (error='', harness='')"}

    monkeypatch.setattr(sandbox_common, "run_golden_episode", no_verdict)

    result = asyncio.run(smoke._run_openenv(tmp_path, "fix-git", backend="e2b", agent=smoke.GOLDEN, base_url=""))
    assert result["reward"] == 0.0
    assert result["exit_status"] == "AgentError"
    assert "no canonical verdict" in result["agent_metrics"]["error"]


def test_openenv_connector_maps_a_timeout_to_time_limit_exceeded(smoke, monkeypatch, tmp_path):
    monkeypatch.setattr(sandbox_common, "load_backend", lambda name: object())
    monkeypatch.setenv("AGENT_TRIAL_TIMEOUT", "0")  # expire immediately, no real wait

    async def hangs(backend, tasks_dir, task_id, **kwargs):
        await asyncio.sleep(10)
        return {"reward": 1.0}

    monkeypatch.setattr(sandbox_common, "run_golden_episode", hangs)

    result = asyncio.run(smoke._run_openenv(tmp_path, "fix-git", backend="e2b", agent=smoke.GOLDEN, base_url=""))
    assert result == {"reward": 0.0, "exit_status": "TimeLimitExceeded", "eval_report": {}, "agent_metrics": {}}


def test_openenv_connector_reports_an_unknown_backend_as_a_clean_exit(smoke, tmp_path):
    with pytest.raises(SystemExit, match="unknown sandbox backend"):
        asyncio.run(smoke._run_openenv(tmp_path, "fix-git", backend="nonesuch", agent=smoke.GOLDEN, base_url=""))
