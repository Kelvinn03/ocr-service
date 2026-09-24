"""Penjaga Dockerfile (port `tests/test_docker_image.py` existing, bagian OCR)."""

import re
from pathlib import Path

from ocr_service.settings import Settings

DOCKERFILE = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text(encoding="utf-8")


def test_warmup_lang_matches_engine_default():
    # lang memilih model yang di-download. Bila warm-up dan engine berbeda, image membawa bobot yang
    # salah dan download terjadi di dalam request pertama.
    warmup = [line for line in DOCKERFILE.splitlines() if "PaddleOCR(" in line]
    assert len(warmup) == 1
    found = re.search(r"lang=['\"]([^'\"]+)['\"]", warmup[0])
    assert found and found.group(1) == Settings(_env_file=None).lang


def test_warmup_fetches_every_optional_model():
    # Flag orientasi/unwarping bisa dinyalakan lewat env tanpa rebuild; modelnya harus sudah ada.
    warmup = next(line for line in DOCKERFILE.splitlines() if "PaddleOCR(" in line)
    for flag in ("use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation"):
        assert f"{flag}=True" in warmup, flag


def test_home_is_pinned_and_cache_copied_to_same_path():
    assert re.search(r"^\s*HOME=/root", DOCKERFILE, re.MULTILINE)
    copies = [ln for ln in DOCKERFILE.splitlines() if ln.strip().startswith("COPY") and "--from=ocr-models" in ln]
    assert copies and copies[0].split()[-2:] == ["/root/.paddlex", "/root/.paddlex"]


def test_single_uvicorn_worker():
    assert '"--workers", "1"' in DOCKERFILE
