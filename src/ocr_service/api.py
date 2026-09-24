"""Route: `POST /v1/ocr`, `GET /healthz`, `GET /readyz`.

Pemetaan status (M2 `HttpOcrClient`: 408/429/5xx = sementara, 4xx lain = permanen):
  400 bad_request · 413 image_too_large · 415 unsupported_media_type · 422 invalid_metadata/invalid_image
  503 busy/not_ready (+ Retry-After) · 500 engine_error

Log hanya request_id, status, error_code, timing, engine. Tidak pernah teks OCR atau isi gambar.
"""

import io
import json
import logging
import time
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import ValidationError

from ocr_service import __version__
from ocr_service.contract import ENGINE_ID, ErrorCode, ErrorResponse, OcrLine, OcrMetadata, OcrResponse
from ocr_service.engine import EngineResult
from ocr_service.pool import EnginePool, PoolBusy, PoolNotReady
from ocr_service.settings import Settings

log = logging.getLogger("ocr_service.api")

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"

router = APIRouter()

_ERROR_RESPONSES = {
    code: {"model": ErrorResponse} for code in (400, 413, 415, 422, 500, 503)
}


class _Reject(Exception):
    def __init__(self, status: int, code: ErrorCode, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclass
class _Outcome:
    result: EngineResult
    engine_version: str
    timing_ms: int
    queue_ms: int


def _error(
    status: int, code: ErrorCode, message: str, request_id: str | None, headers: dict | None = None
) -> JSONResponse:
    body = ErrorResponse(request_id=request_id, error_code=code, message=message)
    return JSONResponse(status_code=status, content=body.model_dump(), headers=headers)


def _decode_png(data: bytes, max_pixels: int) -> Image.Image:
    """Cek ukuran dari header dulu (tanpa decode), baru decode penuh — pagar decompression bomb."""
    try:
        img = Image.open(io.BytesIO(data))
        if img.format != "PNG":
            raise _Reject(415, "unsupported_media_type", "image bukan PNG")
        width, height = img.size
        if width * height > max_pixels:
            raise _Reject(413, "image_too_large", f"image {width}x{height} melebihi {max_pixels} piksel")
        img.load()
    except _Reject:
        raise
    except Image.DecompressionBombError:
        raise _Reject(413, "image_too_large", "image melebihi batas piksel") from None
    except Exception:
        # Pesan exception PIL tidak diteruskan: cukup kodenya.
        raise _Reject(422, "invalid_image", "PNG tidak dapat di-decode") from None
    return img


def _infer(pool: EnginePool, img: Image.Image, timeout: float) -> _Outcome:
    t0 = time.perf_counter()
    with pool.acquire(timeout) as engine:
        t1 = time.perf_counter()
        result = engine.predict(img)
        t2 = time.perf_counter()
        version = engine.version
    return _Outcome(result, version, int((t2 - t1) * 1000), int((t1 - t0) * 1000))


@router.post(
    "/v1/ocr",
    response_model=OcrResponse,
    responses=_ERROR_RESPONSES,
    summary="OCR satu halaman PNG",
)
async def ocr_page(
    request: Request,
    image: Annotated[UploadFile | None, File(description="PNG satu halaman")] = None,
    metadata: Annotated[str | None, Form(description='JSON {"request_id": "..."}')] = None,
) -> JSONResponse:
    settings: Settings = request.app.state.settings
    pool: EnginePool = request.app.state.pool
    started = time.perf_counter()
    request_id: str | None = None

    def done(status: int, code: ErrorCode | None, **extra) -> None:
        log.info(
            "ocr request",
            extra={
                "event": "ocr_request",
                "request_id": request_id,
                "status": status,
                "error_code": code,
                "total_ms": int((time.perf_counter() - started) * 1000),
                **extra,
            },
        )

    try:
        if metadata is None:
            raise _Reject(400, "bad_request", "part 'metadata' wajib ada")
        try:
            raw_meta = json.loads(metadata)
        except ValueError:
            raise _Reject(400, "bad_request", "metadata bukan JSON") from None
        try:
            meta = OcrMetadata.model_validate(raw_meta)
        except ValidationError:
            raise _Reject(422, "invalid_metadata", "metadata tidak sesuai skema (request_id)") from None
        request_id = meta.request_id

        if image is None:
            raise _Reject(400, "bad_request", "part 'image' wajib ada")
        data = await image.read(settings.max_image_bytes + 1)
        if len(data) > settings.max_image_bytes:
            raise _Reject(413, "image_too_large", f"image melebihi {settings.max_image_bytes} byte")
        if not data:
            raise _Reject(400, "bad_request", "image kosong")
        # Magic bytes, bukan Content-Type part: header dari pemanggil tidak dipercaya.
        if not data.startswith(PNG_MAGIC):
            raise _Reject(415, "unsupported_media_type", "image bukan PNG")
        img = await run_in_threadpool(_decode_png, data, settings.max_image_pixels)
    except _Reject as rej:
        done(rej.status, rej.code)
        return _error(rej.status, rej.code, rej.message, request_id)

    retry = {"Retry-After": str(settings.retry_after_s)}
    try:
        # Threadpool: inference bersifat blocking beberapa detik; di event loop ia akan menserialkan
        # yang justru ingin diparalelkan pool, dan probe tidak terjawab selama halaman diproses.
        out = await run_in_threadpool(_infer, pool, img, settings.acquire_timeout_s)
    except PoolBusy:
        done(503, "busy")
        return _error(503, "busy", "semua engine sedang dipakai", request_id, retry)
    except PoolNotReady:
        done(503, "not_ready")
        return _error(503, "not_ready", "engine belum siap", request_id, retry)
    except Exception as exc:
        # 500, bukan 422: gambar sudah lolos validasi, jadi kegagalan ada di engine (OOM GPU, bug
        # paddle) dan bisa hilang saat dicoba ulang. Client M2 memperlakukan 5xx sebagai sementara;
        # "engine selalu gagal pada halaman yang sama" menjadi permanen lewat batas percobaan di M2.
        # Pesan exception tidak masuk body/log (hanya tipe + traceback), tidak ada isi teks di sana.
        log.error(
            "engine error",
            exc_info=True,
            extra={"event": "engine_error", "request_id": request_id, "error_type": type(exc).__name__},
        )
        done(500, "engine_error")
        return _error(500, "engine_error", "engine OCR gagal memproses halaman", request_id)

    lines = [OcrLine(text=ln.text, confidence=ln.confidence, bbox=ln.bbox) for ln in out.result.lines]
    body = OcrResponse(
        request_id=request_id,
        engine=ENGINE_ID,
        engine_version=out.engine_version,
        text="\n".join(ln.text for ln in lines),
        lines=lines,
        rotation_applied=out.result.rotation_applied,  # type: ignore[arg-type]
        timing_ms=out.timing_ms,
    )
    done(
        200,
        None,
        engine=ENGINE_ID,
        engine_version=out.engine_version,
        timing_ms=out.timing_ms,
        queue_ms=out.queue_ms,
        line_count=len(lines),
        dropped_lines=out.result.dropped_lines,
        rotation_applied=out.result.rotation_applied,
    )
    return JSONResponse(content=body.model_dump())


@router.get("/healthz")
def healthz(request: Request) -> JSONResponse:
    """Proses hidup. Gagal (503) hanya bila pool gagal dibangun, agar liveness me-restart pod."""
    pool: EnginePool = request.app.state.pool
    ok = pool.state != "failed"
    return JSONResponse(
        status_code=200 if ok else 503,
        content={"ok": ok, "service_version": __version__, "pool_state": pool.state, "error": pool.error},
    )


@router.get("/readyz")
def readyz(request: Request) -> JSONResponse:
    """Siap menerima halaman: semua engine sudah dibangun. `device` memperlihatkan fallback CPU."""
    pool: EnginePool = request.app.state.pool
    ready = pool.state == "ready"
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "ready": ready,
            "pool_state": pool.state,
            "device": pool.device,
            "pool_size": pool.size,
            "in_use": pool.in_use,
            "engine": ENGINE_ID,
            "engine_version": pool.engine_version,
        },
    )
