"""HTTP API (FastAPI). Everything except /health and the dashboard requires
``Authorization: Bearer $GB10_ASSETS_TOKEN`` (unless GB10_ASSETS_TOKEN=none, then any request is accepted).

    GET    /health                     API version, queue state (no auth, used for the client version check)
    GET    /dashboard                  monitoring page for the browser (no auth, read only)
    GET    /dashboard/data?since=T     its data: system metrics after time T, tools, event log (no auth)
    GET    /tools                      tool names and their parameter JSON schemas
    POST   /jobs                       multipart: field "request" = SubmitJob JSON, files "files" = inputs
    GET    /jobs                       all jobs
    GET    /jobs/{id}                  JobInfo
    GET    /jobs/{id}/files/{name}     one result file
    GET    /jobs/{id}/log              worker log (text)
    DELETE /jobs/{id}                  cancel and delete the job with all files
"""
from __future__ import annotations

import secrets
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, PlainTextResponse
from pydantic import ValidationError

from .. import __version__
from ..shared.schemas import REQUESTS, BlenderScript, Health, JobInfo, SubmitJob, check_file_name
from .config import Settings
from .monitor import Monitor
from .queue import JobQueue
from .runners import TOOLS, install_problem

MAX_UPLOAD_BYTES = 512 * 1024**2
DASHBOARD_HTML = Path(__file__).with_name("dashboard.html")


def create_app(settings: Settings, queue: JobQueue | None = None) -> FastAPI:
    queue = queue or JobQueue(settings)
    monitor = Monitor(settings, queue)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        queue.start()
        monitor.start()
        yield
        monitor.stop()
        queue.shutdown()

    app = FastAPI(title="gb10-assets", version=__version__, lifespan=lifespan)
    app.state.queue = queue

    def auth(request: Request) -> None:
        if settings.token is None:
            return
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token.encode(), settings.token.encode()):
            raise HTTPException(401, "invalid or missing bearer token", headers={"WWW-Authenticate": "Bearer"})

    def job_or_404(job_id: str) -> JobInfo:
        info = queue.get(job_id)
        if info is None:
            raise HTTPException(404, f"job {job_id} not found")
        return info

    @app.get("/health")
    def health() -> Health:
        return Health(version=__version__, **queue.status())

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> str:
        return DASHBOARD_HTML.read_text()

    @app.get("/dashboard/data")
    def dashboard_data(since: float = 0.0) -> dict:
        return monitor.dashboard(since)

    @app.get("/tools", dependencies=[Depends(auth)])
    def tools() -> dict:
        return {name: model.model_json_schema() for name, model in REQUESTS.items() if name in TOOLS}

    @app.post("/jobs", dependencies=[Depends(auth)], status_code=201)
    def submit(request: str = Form(...), files: list[UploadFile] = File(default=[])) -> JobInfo:
        try:
            sub = SubmitJob.model_validate_json(request)
            if sub.tool not in REQUESTS or sub.tool not in TOOLS:
                raise HTTPException(422, f"unknown tool {sub.tool!r}, available: {sorted(TOOLS)}")
            if problem := install_problem(settings, sub.tool):
                raise HTTPException(409, problem)
            params = REQUESTS[sub.tool].model_validate(sub.params)
            names = [check_file_name(f.filename or "") for f in files]
        except ValidationError as e:
            raise HTTPException(422, e.errors(include_url=False, include_context=False)) from None
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        if isinstance(params, BlenderScript) and params.code is not None and not settings.blender_code:
            raise HTTPException(403, "own Blender code is disabled on this server (BLENDER_CODE=0), "
                                     "only the bundled scripts run")
        missing = set(params.input_files()) - set(names)
        if missing:
            raise HTTPException(422, f"input files referenced but not uploaded: {sorted(missing)}")
        job = queue.create(sub.tool, params.model_dump(mode="json"))
        try:
            for f, name in zip(files, names):
                with open(job.path / "in" / name, "wb") as out:
                    copied = 0
                    while chunk := f.file.read(1024**2):
                        copied += len(chunk)
                        if copied > MAX_UPLOAD_BYTES:
                            raise HTTPException(413, f"{name} exceeds {MAX_UPLOAD_BYTES // 1024**2} MB")
                        out.write(chunk)
        except BaseException:
            shutil.rmtree(job.path, ignore_errors=True)
            raise
        return queue.enqueue(job)

    @app.get("/jobs", dependencies=[Depends(auth)])
    def list_jobs() -> list[JobInfo]:
        return queue.list()

    @app.get("/jobs/{job_id}", dependencies=[Depends(auth)])
    def get_job(job_id: str) -> JobInfo:
        return job_or_404(job_id)

    @app.get("/jobs/{job_id}/files/{name}", dependencies=[Depends(auth)])
    def get_file(job_id: str, name: str) -> FileResponse:
        info = job_or_404(job_id)
        if name not in info.outputs:
            raise HTTPException(404, f"{name} is not a result of job {job_id}")
        return FileResponse(queue.path(job_id) / "out" / name, filename=name)

    @app.get("/jobs/{job_id}/log", dependencies=[Depends(auth)], response_class=PlainTextResponse)
    def get_log(job_id: str) -> str:
        job_or_404(job_id)
        log = queue.path(job_id) / "log.txt"
        return log.read_text(errors="replace") if log.exists() else ""

    @app.delete("/jobs/{job_id}", dependencies=[Depends(auth)], status_code=204)
    def delete_job(job_id: str) -> None:
        if not queue.delete(job_id):
            raise HTTPException(404, f"job {job_id} not found")

    return app


def serve(host: str = "0.0.0.0", port: int = 8750) -> None:
    import sys

    import uvicorn

    settings = Settings.from_env()
    if settings.token is None:
        print("WARNING: GB10_ASSETS_TOKEN=none, authentication is disabled: everyone who can reach "
              f"port {port} can submit, read and delete jobs"
              + (", and run own code in Blender (BLENDER_CODE=0 turns that off)" if settings.blender_code else ""),
              file=sys.stderr)
    uvicorn.run(create_app(settings), host=host, port=port)
