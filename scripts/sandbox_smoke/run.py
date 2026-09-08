"""Golden smoke of one (connector, backend, agent, benchmark) combination on the real sandbox API.

PASS iff the verifier returns reward 1.0. What a run proves, the axes,
credentials, and how to choose what to run: README.md next to this file.
"""

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]

# The whole smoke should finish in minutes; a hung platform call must not eat
# hours. Overridable (AGENT_TRIAL_TIMEOUT) for slow first-time template builds.
_DEFAULT_TRIAL_TIMEOUT_S = "1200"

# The connector-neutral name for "execute the task's own reference solution".
GOLDEN = "golden"


@dataclass(frozen=True)
class Benchmark:
    tasks_dir: Callable[[], Path]  # resolved lazily: may fetch on first use
    smoke_task: str  # the instance a smoke run uses when --task is not given


_TB2_REPO = "https://github.com/laude-institute/terminal-bench-2.git"


def _tb2_tasks_dir() -> Path:
    """A Terminal-Bench-2 checkout: TB2_TASKS_DIR if set, else a cached shallow clone.

    TB2 task directories are native Harbor tasks (instruction.md, task.toml
    with a prebuilt official docker_image, tests/, solution/), so a checkout
    is directly usable as HARBOR_TASKS_DIR -- no preparation step.
    """
    env = os.environ.get("TB2_TASKS_DIR", "").strip()
    if env:
        return Path(env)
    cache = Path.home() / ".cache" / "miles-sandbox-smoke" / "terminal-bench-2"
    if not (cache / "fix-git").is_dir():
        print(f"sandbox-smoke: cloning {_TB2_REPO} -> {cache}", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        # clone into a scratch dir and rename into place, so an interrupted
        # clone leaves nothing at the final path to block the next run
        scratch = cache.with_name(cache.name + ".partial")
        shutil.rmtree(cache, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)
        subprocess.run(["git", "clone", "--depth", "1", _TB2_REPO, str(scratch)], check=True)
        scratch.rename(cache)
    return cache


BENCHMARKS: dict[str, Benchmark] = {
    # fix-git: small, easy, the task our golden runs have always used.
    "tb2": Benchmark(tasks_dir=_tb2_tasks_dir, smoke_task="fix-git"),
}


async def _run_harbor(tasks_dir: Path, task: str, *, backend: str, agent: str, base_url: str) -> dict[str, Any]:
    """One trial through examples/experimental/harbor/harbor_agent_function."""
    sys.path.insert(0, str(REPO))  # for miles.rollout.agentic; no PYTHONPATH needed
    sys.path.insert(0, str(REPO / "examples" / "experimental" / "harbor"))
    try:
        import harbor  # noqa: F401 -- probed eagerly: harbor_agent_function defers its harbor imports
        import harbor_agent_function as haf
    except ImportError as e:
        raise SystemExit(
            f"{e}\nharbor is not importable in this environment; the install line is in {Path(__file__).parent / 'README.md'}"
        ) from e

    if agent == GOLDEN:
        agent = "oracle"  # harbor's solution-executing agent
    os.environ["HARBOR_TASKS_DIR"] = str(tasks_dir)
    os.environ["HARBOR_ENV_TYPE"] = backend
    os.environ.setdefault("AGENT_TRIAL_TIMEOUT", _DEFAULT_TRIAL_TIMEOUT_S)
    return await haf.run(
        base_url=base_url,
        prompt=[],
        request_kwargs={},
        metadata={"instance_id": task, "agent_name": agent},
    )


async def _run_openenv(tasks_dir: Path, task: str, *, backend: str, agent: str, base_url: str) -> dict[str, Any]:
    """One golden replay through examples/experimental/openenv/openenv_sandbox_common.

    Only the golden path is wired here: a harness episode goes through
    openenv_agent_function.run_for_training, whose exit_status vocabulary
    (pre-#2802: "completed" / "timeout") this driver's PASS check does not
    recognize yet -- wire it once #2802 lands and openenv_agent_function speaks
    "Submitted" the way harbor_agent_function already does.
    """
    if agent != GOLDEN:
        raise NotImplementedError(
            f"the openenv connector only runs {GOLDEN!r} for now: a harness episode needs "
            "#2802's exit_status vocabulary in openenv_agent_function first"
        )

    openenv_dir = REPO / "examples" / "experimental" / "openenv"
    sys.path.insert(0, str(openenv_dir))
    try:
        import openenv_sandbox_common as sandbox_common
    except ImportError as e:
        raise SystemExit(f"{e}\nopenenv is not importable in this environment; see {openenv_dir / 'README.md'}") from e

    try:
        sandbox_backend = sandbox_common.load_backend(backend)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    except ImportError as e:
        raise SystemExit(
            f"{e}\nthe {backend} sandbox SDK is not importable in this environment; see {openenv_dir / 'README.md'}"
        ) from e

    os.environ["OPENENV_TB2_TASKS_DIR"] = str(tasks_dir)
    trial_timeout_s = int(os.environ.get("AGENT_TRIAL_TIMEOUT", _DEFAULT_TRIAL_TIMEOUT_S))
    try:
        m = await asyncio.wait_for(
            sandbox_common.run_golden_episode(sandbox_backend, tasks_dir, task), timeout=trial_timeout_s
        )
    except asyncio.TimeoutError:
        return _openenv_failed("TimeLimitExceeded")

    reward = m.get("reward")
    metrics = {k: v for k, v in m.items() if k != "reward"}
    if reward is None:
        return _openenv_result(0.0, "AgentError", metrics)
    return _openenv_result(float(reward), "Submitted", metrics)


def _openenv_result(reward: float, exit_status: str, agent_metrics: dict[str, Any]) -> dict[str, Any]:
    """The schema every connector in this driver returns."""
    return {"reward": reward, "exit_status": exit_status, "eval_report": {}, "agent_metrics": agent_metrics}


def _openenv_failed(exit_status: str) -> dict[str, Any]:
    return _openenv_result(0.0, exit_status, {})


# connector name -> (tasks_dir, task, backend=, agent=, base_url=) -> result dict
CONNECTORS: dict[str, Callable[..., Any]] = {
    "harbor": _run_harbor,
    "openenv": _run_openenv,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--connector", required=True, choices=sorted(CONNECTORS))
    parser.add_argument(
        "--backend", required=True, help="sandbox platform, passed through to the connector (e2b, daytona, modal, ...)"
    )
    parser.add_argument(
        "--agent",
        default=GOLDEN,
        help=f"{GOLDEN!r} (default, no model needed) or a real harness name (needs --base-url)",
    )
    parser.add_argument("--benchmark", default="tb2", choices=sorted(BENCHMARKS))
    parser.add_argument(
        "--base-url", default="", help="OpenAI-compatible endpoint; required for any agent other than the golden one"
    )
    parser.add_argument("--task", default="", help="override the benchmark's preset smoke instance")
    parser.add_argument("--tasks-dir", type=Path, default=None, help="override the benchmark's task directory")
    args = parser.parse_args()

    if args.agent != GOLDEN and not args.base_url:
        parser.error(
            f"--agent {args.agent} is a real harness and needs --base-url (only {GOLDEN!r} runs without a model)"
        )
    base_url = args.base_url or "http://smoke.invalid/sessions/smoke"  # the golden agent never calls it
    benchmark = BENCHMARKS[args.benchmark]
    tasks_dir = args.tasks_dir if args.tasks_dir is not None else benchmark.tasks_dir()
    task = args.task or benchmark.smoke_task

    print(
        f"sandbox-smoke: connector={args.connector} backend={args.backend} agent={args.agent} "
        f"benchmark={args.benchmark} task={task}",
        flush=True,
    )
    result = asyncio.run(
        CONNECTORS[args.connector](tasks_dir, task, backend=args.backend, agent=args.agent, base_url=base_url)
    )
    print(f"sandbox-smoke: result={result}", flush=True)

    reward = float(result.get("reward", 0.0))
    exit_status = result.get("exit_status", "")
    if reward == 1.0 and exit_status == "Submitted":
        print("sandbox-smoke: PASS", flush=True)
        return 0
    print(f"sandbox-smoke: FAIL (reward={reward}, exit_status={exit_status!r})", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
