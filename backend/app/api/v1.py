from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from .. import planner, store
from .. import render as renderer
from ..config import settings
from ..schemas import Delivery, GenerateRequest, Rating, SegmentEdit

ACTIVE = {"analysing", "planning", "writing", "queued", "synthesizing", "rewriting", "modulating", "mixing"}

router = APIRouter(prefix="/api/v1")


@router.get("/health")
def health():
    return {"ok": True, "mock": settings.use_mock, "llm": "ollama", "tts": "indic-parler-tts"}


@router.post("/performances", status_code=201)
def create_performance(req: GenerateRequest, background: BackgroundTasks):
    pid = store.create(req)
    background.add_task(planner.run_script_job, pid)
    return {"id": pid, "status": store.read(pid, "status.json")}


@router.get("/performances")
def list_performances():
    return [{"id": pid, "request": store.read(pid, "request.json")} for pid in store.list_ids()]


def _get_or_404(pid: str) -> dict:
    request = store.read(pid, "request.json")
    if request is None:
        raise HTTPException(404, "performance not found")
    return request


@router.get("/performances/{pid}")
def get_performance(pid: str):
    return {
        "id": pid,
        "request": _get_or_404(pid),
        "script": store.read(pid, "script.json"),
        "timeline": store.read(pid, "timeline.json"),
        "credits": store.read(pid, "credits.json") or [],
        "mock": ((store.read(pid, "meta.json") or {}).get("settings") or {}).get("mock", False),
        "status": store.read(pid, "status.json"),
    }


@router.get("/performances/{pid}/status")
def get_status(pid: str):
    _get_or_404(pid)
    return store.read(pid, "status.json")


@router.post("/performances/{pid}/render", status_code=202)
def render(pid: str, background: BackgroundTasks):
    _get_or_404(pid)
    status = store.read(pid, "status.json")
    if store.read(pid, "script.json") is None or status["stage"] in ACTIVE:
        raise HTTPException(409, f"cannot render while stage is '{status['stage']}'")
    queued = store.write_status(pid, "queued", progress=0.0)
    background.add_task(renderer.run_render_job, pid)
    return queued


@router.patch("/performances/{pid}/segments/{sid}")
def edit_segment(pid: str, sid: str, edit: SegmentEdit):
    _get_or_404(pid)
    script = store.read(pid, "script.json")
    status = store.read(pid, "status.json")
    if script is None or status["stage"] in ACTIVE:
        raise HTTPException(409, f"cannot edit while stage is '{status['stage']}'")
    seg = next((s for s in script["segments"] if s["id"] == sid), None)
    if seg is None:
        raise HTTPException(404, "segment not found")
    if edit.text is not None:
        seg["text"] = edit.text
    if edit.delivery is not None:
        seg["delivery"] = Delivery.model_validate({**seg["delivery"], **edit.delivery}).model_dump(mode="json")
    v = store.save_script(pid, script, f"user edit: {sid}")
    store.write_status(pid, status["stage"], status["detail"], status["progress"])
    return {"script_version": v, "segment": seg}


@router.get("/performances/{pid}/audio")
def get_audio(pid: str, fmt: Literal["mp3", "wav"] = "mp3"):
    _get_or_404(pid)
    path = settings.performances_dir / pid / f"performance.{fmt}"
    if not path.exists():
        raise HTTPException(404, f"no {fmt} audio yet")
    return FileResponse(path, media_type="audio/mpeg" if fmt == "mp3" else "audio/wav")


@router.get("/performances/{pid}/clips/{sid}")
def get_clip(pid: str, sid: str):
    _get_or_404(pid)
    path = settings.performances_dir / pid / "clips" / f"{sid}.wav"
    if not path.resolve().is_relative_to(path.parent.resolve()) or not path.exists():
        raise HTTPException(404, "clip not found")
    return FileResponse(path, media_type="audio/wav")


@router.post("/performances/{pid}/ratings", status_code=201)
def add_rating(pid: str, rating: Rating):
    _get_or_404(pid)
    version = (store.read(pid, "status.json") or {}).get("script_version")
    store.append_rating({**rating.model_dump(), "performance_id": pid, "script_version": version})
    return {"ok": True}
