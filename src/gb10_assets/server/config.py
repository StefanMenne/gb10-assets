"""Server settings from environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
NO_TOKEN = "none"  # GB10_ASSETS_TOKEN=none: no authentication, every request is accepted


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() not in ("0", "false", "no", "off")


@dataclass
class Settings:
    token: str | None  # None: authentication disabled
    jobs_dir: Path = REPO_ROOT / "jobs"
    # installed tools (scripts/install.sh): <install_dir>/<tool>/env (uv environment) and .../weights
    install_dir: Path = REPO_ROOT / "installed"   # same default as scripts/lib/config.sh
    tools_dir: Path = REPO_ROOT / "tools"         # tool definitions: run.py, pyproject.toml, download.sh, ...
    worker_idle_s: float = 0.0       # keep an idle worker this long for a following job of the same tool
    memory_wait_s: float = 600.0     # wait this long for enough free memory before failing a job
    worker_ready_s: float = 1800.0   # max model loading time
    job_ttl_h: float = 24.0          # finished jobs are deleted after this time if not fetched
    blender_code: bool = True        # blender tool: also run own code, not only the bundled scripts
    # dashboard (no authentication): show job parameters (prompts cut to 160 characters, never own code)
    dashboard_details: bool = False

    @classmethod
    def from_env(cls) -> Settings:
        token: str | None = os.environ.get("GB10_ASSETS_TOKEN", "")
        if token.lower() == NO_TOKEN:
            token = None
        elif len(token) < 16:
            raise SystemExit("GB10_ASSETS_TOKEN must be set to a random string of at least 16 characters "
                             f"(e.g. `openssl rand -hex 32`), or to {NO_TOKEN!r} to disable authentication")
        return cls(
            token=token,
            jobs_dir=Path(os.environ.get("JOBS_DIR", REPO_ROOT / "jobs")).resolve(),
            install_dir=Path(os.environ.get("INSTALL_DIR", REPO_ROOT / "installed")).resolve(),
            tools_dir=Path(os.environ.get("TOOLS_DIR", REPO_ROOT / "tools")).resolve(),
            worker_idle_s=float(os.environ.get("WORKER_IDLE_S", 0)),
            memory_wait_s=float(os.environ.get("MEMORY_WAIT_S", 600)),
            worker_ready_s=float(os.environ.get("WORKER_READY_S", 1800)),
            job_ttl_h=float(os.environ.get("JOB_TTL_H", 24)),
            blender_code=_flag("BLENDER_CODE", "1"),
            dashboard_details=_flag("DASHBOARD_DETAILS", "0"),
        )
