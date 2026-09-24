"""App factory. Jalankan: `uvicorn ocr_service.main:create_app --factory --port 8002 --workers 1`.

Satu worker proses: konkurensi ada DI DALAM proses (pool engine berbagi satu CUDA context). Worker
kedua membayar CUDA context sendiri (~300-500MB) dan tidak bisa berbagi puncak transient antar halaman.
"""

import json
import logging
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ocr_service import __version__
from ocr_service.api import router
from ocr_service.contract import ErrorResponse
from ocr_service.engine import EngineFactory, PaddleEngineFactory
from ocr_service.pool import EnginePool
from ocr_service.settings import Settings, get_settings

# Hanya field ini yang keluar ke log. Menambah field = keputusan sadar (tidak ada teks OCR/gambar).
_LOG_FIELDS = (
    "event", "request_id", "status", "error_code", "error_type", "total_ms", "timing_ms", "queue_ms",
    "engine", "engine_version", "line_count", "dropped_lines", "rotation_applied", "device", "pool_size",
)  # fmt: skip

# Overhead multipart di atas batas byte gambar (boundary, header part, metadata).
_MULTIPART_OVERHEAD = 1024 * 1024


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = {"ts": self.formatTime(record), "level": record.levelname, "logger": record.name}
        out["msg"] = record.getMessage()
        out.update({k: getattr(record, k) for k in _LOG_FIELDS if getattr(record, k, None) is not None})
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out, ensure_ascii=False)


def configure_logging(level: str) -> None:
    logger = logging.getLogger("ocr_service")
    logger.setLevel(level.upper())
    if not any(getattr(h, "_ocr_service", False) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        handler._ocr_service = True  # type: ignore[attr-defined]
        logger.addHandler(handler)


def create_app(
    settings: Settings | None = None,
    engine_factory: EngineFactory | None = None,
    *,
    background_build: bool = True,
) -> FastAPI:
    """`engine_factory` diganti di test (FakeEngine) agar tidak butuh paddle.

    `background_build=True`: pool dibangun di thread latar, jadi `/healthz` langsung menjawab selama
    paddle/CUDA dimuat (bisa beberapa menit) dan `/readyz` baru 200 setelah semua engine siap.
    """
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    pool = EnginePool(engine_factory or PaddleEngineFactory(settings), settings.pool_size_for)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if background_build:
            threading.Thread(target=pool.build, name="ocr-pool-build", daemon=True).start()
        else:
            pool.build()
        yield

    app = FastAPI(title="ocr-service", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.pool = pool

    @app.middleware("http")
    async def reject_oversized(request: Request, call_next):
        # Tolak sebelum body dibaca bila Content-Length sudah jelas melebihi batas.
        length = request.headers.get("content-length")
        if request.url.path == "/v1/ocr" and length and length.isdigit():
            if int(length) > settings.max_image_bytes + _MULTIPART_OVERHEAD:
                logging.getLogger("ocr_service.api").info(
                    "ocr request", extra={"event": "ocr_request", "status": 413, "error_code": "image_too_large"}
                )
                body = ErrorResponse(request_id=None, error_code="image_too_large", message="request terlalu besar")
                return JSONResponse(status_code=413, content=body.model_dump())
        return await call_next(request)

    app.include_router(router)
    return app
