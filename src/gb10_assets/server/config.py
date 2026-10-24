"""Server settings from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class Settings:
    token: str
    jobs_dir: Path = REPO_ROOT / "jobs"
    models_dir: Path = REPO_ROOT / "models"   # same default as scripts/download-all.sh
    tools_dir: Path = REPO_ROOT / "tools"
    # where the per-tool uv environments live: <tool_envs_dir>/<tool>/.venv (Docker: /opt); default tools/<tool>/.venv
    tool_envs_dir: Path | None = None
    worker_idle_s: float = 0.0       # keep an idle worker this long for a following job of the same tool
    memory_wait_s: float = 600.0     # wait this long for enough free memory before failing a job
    worker_ready_s: float = 1800.0   # max model loading time
    job_ttl_h: float = 24.0          # finished jobs are deleted after this time if not fetched

    @classmethod
    def from_env(cls) -> Settings:
        token = os.environ.get("GB10_ASSETS_TOKEN", "")
        if len(token) < 16:
            raise SystemExit("GB10_ASSETS_TOKEN must be set to a random string of at least 16 characters "
                             "(e.g. `openssl rand -hex 32`)")
        envs = os.environ.get("TOOL_ENVS_DIR")
        return cls(
            token=token,
            jobs_dir=Path(os.environ.get("JOBS_DIR", REPO_ROOT / "jobs")).resolve(),
            models_dir=Path(os.environ.get("MODELS_DIR", REPO_ROOT / "models")).resolve(),
            tools_dir=Path(os.environ.get("TOOLS_DIR", REPO_ROOT / "tools")).resolve(),
            tool_envs_dir=Path(envs).resolve() if envs else None,
            worker_idle_s=float(os.environ.get("WORKER_IDLE_S", 0)),
            memory_wait_s=float(os.environ.get("MEMORY_WAIT_S", 600)),
            worker_ready_s=float(os.environ.get("WORKER_READY_S", 1800)),
            job_ttl_h=float(os.environ.get("JOB_TTL_H", 24)),
        )
