"""FakeEngine + app factory: semua test berjalan tanpa paddle."""

import io
import json
from collections.abc import Callable, Iterator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from ocr_service.engine import EngineLine, EngineResult
from ocr_service.main import create_app
from ocr_service.settings import Settings

# Teks sintetis yang dipakai untuk memastikan isi OCR tidak pernah masuk log.
SECRET_TEXT = "NAMA RAHASIA 3171234567890001"


class FakeEngine:
    engine_id = "paddleocr"
    version = "0.0-fake"

    def __init__(self, result: EngineResult | Callable[[Image.Image], EngineResult]):
        self._result = result
        self.calls = 0

    def predict(self, image: Image.Image) -> EngineResult:
        self.calls += 1
        return self._result(image) if callable(self._result) else self._result


class FakeFactory:
    def __init__(self, result, device: str = "cpu"):
        self._result = result
        self._device = device
        self.engines: list[FakeEngine] = []

    def device(self) -> str:
        return self._device

    def create(self) -> FakeEngine:
        engine = FakeEngine(self._result)
        self.engines.append(engine)
        return engine


def default_result() -> EngineResult:
    return EngineResult(
        lines=[
            EngineLine(text="SERTIFIKAT HAK GUNA BANGUNAN", confidence=0.97, bbox=[112, 80, 890, 132]),
            EngineLine(text=SECRET_TEXT, confidence=0.58123456, bbox=[112, 150, 540, 188]),
        ]
    )


def make_png(width: int = 40, height: int = 20) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (width, height), "white").save(buf, format="PNG")
    return buf.getvalue()


def post_ocr(client: TestClient, png: bytes | None, request_id: str | None = "job_x:page-1", **kw):
    """Kirim persis seperti `HttpOcrClient` M2 (multipart image + metadata JSON)."""
    files: dict = {}
    if png is not None:
        files["image"] = ("page.png", png, "image/png")
    if request_id is not None:
        files["metadata"] = (None, json.dumps({"request_id": request_id}), "application/json")
    return client.post("/v1/ocr", files=files, **kw)


@pytest.fixture
def settings() -> Settings:
    return Settings(_env_file=None, cpu_pool_size=1, acquire_timeout_s=0.2, retry_after_s=3)


@pytest.fixture
def make_client(settings) -> Iterator[Callable[..., TestClient]]:
    clients: list[TestClient] = []

    def _make(result=None, settings_override: Settings | None = None, factory=None) -> TestClient:
        factory = factory or FakeFactory(result if result is not None else default_result())
        app = create_app(settings_override or settings, factory, background_build=False)
        client = TestClient(app)
        client.__enter__()  # jalankan lifespan (pool dibangun sinkron)
        clients.append(client)
        return client

    yield _make
    for c in clients:
        c.__exit__(None, None, None)


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
