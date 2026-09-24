"""Kontrak `POST /v1/ocr` — satu-satunya sumber kebenaran.

`contract/*.json` di-generate dari modul ini (`python -m ocr_service.export_contract`), dan test gagal
bila file tersebut basi. Perubahan dalam v1 hanya boleh aditif (field baru opsional); perubahan yang
merusak = `/v2` dijalankan berdampingan.

Request (multipart/form-data):
  image    → PNG satu halaman
  metadata → JSON `OcrMetadata`
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

API_VERSION = "v1"
ENGINE_ID = "paddleocr"

# request_id dipakai untuk menelusuri log di kedua sisi, jadi karakternya dibatasi: tidak ada spasi
# (mencegah teks bebas/PII dan log injection). M2 mengirim `<job_id>:page-<n>`.
REQUEST_ID_PATTERN = r"^[A-Za-z0-9._:\-]{1,200}$"


class OcrMetadata(BaseModel):
    """Bagian `metadata` dari request."""

    model_config = ConfigDict(extra="ignore", title="OcrMetadata")

    request_id: str = Field(
        pattern=REQUEST_ID_PATTERN,
        description="ID penelusuran dari pemanggil, di-echo di respons. Tidak boleh memuat data pribadi.",
    )


class OcrLine(BaseModel):
    model_config = ConfigDict(title="OcrLine")

    text: str
    confidence: float = Field(ge=0.0, le=1.0, description="Skor pengenalan baris dari engine, apa adanya.")
    bbox: list[int] = Field(
        min_length=4,
        max_length=4,
        description="[x0, y0, x1, y1] dalam piksel PNG yang dikirim (lihat README: rotation_applied).",
    )


class OcrResponse(BaseModel):
    """Respons 200. Hasil kosong (`text=""`, `lines=[]`) adalah hasil sah, bukan error."""

    model_config = ConfigDict(title="OcrResponse")

    request_id: str
    engine: str = Field(description="Selalu 'paddleocr' (item 15).")
    engine_version: str = Field(description="paddleocr.__version__; dicatat M2 di `versions`.")
    text: str = Field(description="Baris digabung dengan '\\n' sesuai urutan baca engine.")
    lines: list[OcrLine]
    rotation_applied: Literal[0, 90, 180, 270] = Field(
        description="Rotasi halaman (derajat) yang diterapkan engine sebelum deteksi; 0 bila tidak ada."
    )
    timing_ms: int = Field(ge=0, description="Durasi inference di engine (tanpa waktu antre).")


ErrorCode = Literal[
    "bad_request",  # 400: part hilang, metadata bukan JSON, gambar kosong
    "invalid_metadata",  # 422: metadata tidak sesuai skema
    "invalid_image",  # 422: magic bytes PNG tetapi gagal di-decode
    "image_too_large",  # 413: melebihi batas byte/piksel
    "unsupported_media_type",  # 415: bukan PNG
    "busy",  # 503: semua engine terpakai sampai acquire timeout
    "not_ready",  # 503: pool belum/tidak selesai dibangun
    "engine_error",  # 500: engine melempar exception pada gambar valid
]


class ErrorResponse(BaseModel):
    """Body semua respons non-2xx. Tidak pernah memuat isi teks OCR maupun isi gambar."""

    model_config = ConfigDict(title="OcrErrorResponse")

    request_id: str | None = Field(description="Di-echo bila metadata sudah terbaca.")
    error_code: ErrorCode
    message: str
