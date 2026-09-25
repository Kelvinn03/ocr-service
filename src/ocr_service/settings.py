"""Satu-satunya pembaca env. Nama env mempertahankan nama existing (`OCR_ENGINE_POOL_SIZE`, dst.)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OCR_", env_file=".env", extra="ignore")

    # --- pool --------------------------------------------------------------------------------
    # auto = tanya wheel paddle yang terpasang (GPU build + kartu terlihat → gpu, selain itu cpu).
    device: Literal["auto", "gpu", "cpu"] = "auto"
    # Ukuran pool di GPU. Default kode 2 (sama dengan existing); manifest k8s memasang 6.
    engine_pool_size: int = Field(2, ge=1)
    # Di CPU satu engine sudah memakai semua core; lebih dari 1 tidak mempercepat (diukur di existing).
    cpu_pool_size: int = Field(1, ge=1)
    # Batas tunggu engine kosong. Lewat dari ini → 503 + Retry-After (existing: menunggu tanpa batas).
    acquire_timeout_s: float = Field(30.0, gt=0)
    retry_after_s: int = Field(5, ge=1)

    # --- input -------------------------------------------------------------------------------
    # Halaman A4 250 DPI ≈ 2067x2923 px (≈6 MP), PNG beberapa MB. Batas ini longgar, hanya pagar.
    max_image_bytes: int = Field(50 * 1024 * 1024, ge=1)
    max_image_pixels: int = Field(50_000_000, ge=1)

    # --- PaddleOCR ---------------------------------------------------------------------------
    # lang harus sama dengan warm-up di Dockerfile (tests/test_dockerfile.py menjaga ini).
    lang: str = "en"
    # oneDNN gagal di paddlepaddle 3.3.1 / paddleocr 3.7.0 untuk semua halaman (diukur 2026-09-22 di
    # existing). No-op di GPU; di CPU harus tetap False.
    enable_mkldnn: bool = False
    # None = tidak dikirim ke PaddleOCR → default PaddleOCR 3.x, SAMA dengan existing (existing tidak
    # memasang flag ini). Catatan item 24: bila doc orientation / unwarping aktif (default), bbox
    # mengacu ke gambar hasil pra-proses, bukan PNG yang dikirim — traceback scan bisa sedikit bergeser.
    use_doc_orientation_classify: bool | None = None
    use_doc_unwarping: bool | None = None
    use_textline_orientation: bool | None = None

    log_level: str = "INFO"

    def pool_size_for(self, device: str) -> int:
        return self.engine_pool_size if device == "gpu" else self.cpu_pool_size


@lru_cache
def get_settings() -> Settings:
    return Settings()
