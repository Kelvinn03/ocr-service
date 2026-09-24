"""`POST /v1/ocr`: kontrak, kode status, echo request_id, hasil kosong, 503."""

import logging

import pytest

from ocr_service.contract import OcrResponse
from ocr_service.engine import EngineResult
from ocr_service.settings import Settings

from .conftest import SECRET_TEXT, FakeFactory, make_png, post_ocr


def test_ok_matches_contract_and_echoes_request_id(client):
    resp = post_ocr(client, make_png(), "job_abc:page-7")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {
        "request_id", "engine", "engine_version", "text", "lines", "rotation_applied", "timing_ms",
    }  # fmt: skip
    OcrResponse.model_validate(body)
    assert body["request_id"] == "job_abc:page-7"
    assert (body["engine"], body["engine_version"]) == ("paddleocr", "0.0-fake")
    assert body["text"] == "SERTIFIKAT HAK GUNA BANGUNAN\n" + SECRET_TEXT
    assert body["lines"][1] == {"text": SECRET_TEXT, "confidence": 0.58123456, "bbox": [112, 150, 540, 188]}
    assert body["rotation_applied"] == 0
    assert isinstance(body["timing_ms"], int) and body["timing_ms"] >= 0
    # Statistik confidence sengaja tidak ada (dihitung M2).
    assert not {"avg_confidence", "min_confidence", "low_conf_line_count"} & set(body)


def test_empty_result_is_200_not_an_error(make_client):
    client = make_client(EngineResult())
    resp = post_ocr(client, make_png())
    assert resp.status_code == 200
    assert resp.json()["text"] == "" and resp.json()["lines"] == []


def test_rotation_is_reported(make_client):
    client = make_client(EngineResult(rotation_applied=90))
    assert post_ocr(client, make_png()).json()["rotation_applied"] == 90


def test_engine_receives_decoded_image(make_client):
    seen = {}

    def result(img):
        seen["size"] = img.size
        return EngineResult()

    post_ocr(make_client(result), make_png(33, 17))
    assert seen["size"] == (33, 17)


@pytest.mark.parametrize(
    ("png", "request_id", "status", "code"),
    [
        (b"GIF89a....", "r1", 415, "unsupported_media_type"),
        (b"\xff\xd8\xff\xe0jpeg", "r1", 415, "unsupported_media_type"),
        (b"\x89PNG\r\n\x1a\n" + b"garbage" * 10, "r1", 422, "invalid_image"),
        (b"", "r1", 400, "bad_request"),
        (None, "r1", 400, "bad_request"),
        (make_png(), None, 400, "bad_request"),
        (make_png(), "has space", 422, "invalid_metadata"),
        (make_png(), "", 422, "invalid_metadata"),
    ],
)
def test_bad_input_is_permanent_4xx(client, png, request_id, status, code):
    resp = post_ocr(client, png, request_id)
    assert resp.status_code == status
    assert resp.json()["error_code"] == code


def test_metadata_not_json_is_400(client):
    files = {"image": ("p.png", make_png(), "image/png"), "metadata": (None, "{not json", "application/json")}
    resp = client.post("/v1/ocr", files=files)
    assert resp.status_code == 400 and resp.json()["error_code"] == "bad_request"


def test_metadata_without_request_id_is_422(client):
    files = {"image": ("p.png", make_png(), "image/png"), "metadata": (None, "{}", "application/json")}
    assert client.post("/v1/ocr", files=files).status_code == 422


def test_error_echoes_request_id_once_known(client):
    body = post_ocr(client, b"nope", "job_q:page-1").json()
    assert body["request_id"] == "job_q:page-1"


def test_too_many_bytes_is_413(make_client, settings):
    png = make_png(200, 200)
    client = make_client(settings_override=settings.model_copy(update={"max_image_bytes": len(png) - 1}))
    resp = post_ocr(client, png)
    assert resp.status_code == 413 and resp.json()["error_code"] == "image_too_large"


def test_too_many_pixels_is_413(make_client, settings):
    client = make_client(settings_override=settings.model_copy(update={"max_image_pixels": 99}))
    resp = post_ocr(client, make_png(10, 10))
    assert resp.status_code == 413 and resp.json()["error_code"] == "image_too_large"


def test_oversized_content_length_is_rejected_before_reading_body(make_client, settings):
    client = make_client(settings_override=settings.model_copy(update={"max_image_bytes": 10}))
    big = b"\x89PNG\r\n\x1a\n" + b"\0" * (1024 * 1024 + 100)
    resp = post_ocr(client, big)
    # request_id None: middleware menolak sebelum multipart di-parse.
    assert resp.status_code == 413 and resp.json()["request_id"] is None


def test_pool_timeout_is_503_with_retry_after(client):
    pool = client.app.state.pool
    with pool.acquire(1.0):  # pool_size=1: satu-satunya engine dipegang test
        resp = post_ocr(client, make_png(), "job_busy:page-1")
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "3"
    assert resp.json() == {
        "request_id": "job_busy:page-1",
        "error_code": "busy",
        "message": "semua engine sedang dipakai",
    }
    # Engine kembali ke pool: request berikutnya berhasil.
    assert post_ocr(client, make_png()).status_code == 200


def test_engine_exception_is_500(make_client, caplog):
    def boom(img):
        raise RuntimeError(f"cuda oom while reading {SECRET_TEXT}")

    client = make_client(boom)
    with caplog.at_level(logging.INFO, logger="ocr_service"):
        resp = post_ocr(client, make_png())
    assert resp.status_code == 500 and resp.json()["error_code"] == "engine_error"
    assert SECRET_TEXT not in resp.text
    # Engine dikembalikan ke pool walau error.
    assert client.app.state.pool.in_use == 0


def test_pool_build_failure_is_not_ready(make_client, settings):
    class Broken(FakeFactory):
        def create(self):
            raise MemoryError("vram")

    client = make_client(factory=Broken(EngineResult()))
    assert client.get("/readyz").status_code == 503
    assert client.get("/healthz").status_code == 503
    resp = post_ocr(client, make_png())
    assert resp.status_code == 503 and resp.json()["error_code"] == "not_ready"


def test_health_and_ready(client):
    assert client.get("/healthz").json()["ok"] is True
    ready = client.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["pool_size"] == 1 and ready.json()["device"] == "cpu"


def test_gpu_pool_size_used_on_gpu(make_client):
    s = Settings(_env_file=None, engine_pool_size=3, cpu_pool_size=1)
    factory = FakeFactory(EngineResult(), device="gpu")
    client = make_client(settings_override=s, factory=factory)
    assert client.get("/readyz").json()["pool_size"] == 3 and len(factory.engines) == 3


def test_no_ocr_text_in_logs(make_client, caplog):
    client = make_client()
    with caplog.at_level(logging.DEBUG):
        assert post_ocr(client, make_png(), "job_log:page-1").status_code == 200
        post_ocr(client, b"bad", "job_log:page-2")
    records = [r for r in caplog.records if r.name.startswith("ocr_service")]
    assert any(getattr(r, "request_id", None) == "job_log:page-1" for r in records)
    for r in caplog.records:
        blob = r.getMessage() + " " + " ".join(str(v) for v in vars(r).values())
        assert SECRET_TEXT not in blob and "SERTIFIKAT" not in blob


def test_background_build_serves_healthz_then_becomes_ready(settings):
    import threading
    import time

    from fastapi.testclient import TestClient

    from ocr_service.main import create_app

    gate = threading.Event()

    class Slow(FakeFactory):
        def create(self):
            gate.wait(5)
            return super().create()

    app = create_app(settings, Slow(EngineResult()), background_build=True)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code == 503
        assert post_ocr(client, make_png()).json()["error_code"] == "not_ready"
        gate.set()
        for _ in range(100):
            if client.get("/readyz").status_code == 200:
                break
            time.sleep(0.01)
        assert client.get("/readyz").status_code == 200
        assert post_ocr(client, make_png()).status_code == 200
