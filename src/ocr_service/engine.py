"""Engine OCR: protokol + PaddleEngine (port `ocr/engine_paddle.py` + `ocr/pool.py` existing).

Perbedaan dari existing:
- Tidak menghitung statistik confidence (mean/min/below floor): itu aturan M2, dihitung dari `lines`.
- Kunci baris `confidence`/`bbox` (existing `score`/`box`), confidence tidak dibulatkan.
- Flag doc orientation / unwarping / textline orientation dipasang eksplisit (lihat settings.py).
- Fallback API PaddleOCR 2.x (`ocr()`) dibuang: versi dipin ke 3.7.0.
- Print diagnostik `[ocr-pool]` dibuang.

`paddle`/`paddleocr` hanya di-import di dalam fungsi, sehingga paket ini dapat di-import dan dites
tanpa paddle terpasang.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from PIL import Image

from ocr_service.contract import ENGINE_ID
from ocr_service.settings import Settings

log = logging.getLogger("ocr_service.engine")

_ROTATIONS = (0, 90, 180, 270)


@dataclass(frozen=True)
class EngineLine:
    text: str
    confidence: float
    bbox: list[int]


@dataclass
class EngineResult:
    lines: list[EngineLine] = field(default_factory=list)
    rotation_applied: int = 0
    # Baris yang dibuang karena engine tidak memberi skor/posisi. Hanya jumlah yang di-log.
    dropped_lines: int = 0


class Engine(Protocol):
    """Satu instance engine. Tidak thread-safe: pool menjamin satu pemakai pada satu waktu."""

    engine_id: str
    version: str

    def predict(self, image: Image.Image) -> EngineResult: ...


class EngineFactory(Protocol):
    def device(self) -> str: ...

    def create(self) -> Engine: ...


# --- parsing hasil PaddleOCR 3.x (murni, dites tanpa paddle) -----------------------------------


def to_box(value: Any) -> list[int] | None:
    """Normalisasi box/polygon PaddleOCR ke [x0, y0, x1, y1] int.

    rec_boxes sudah persegi sejajar sumbu; rec_polys/dt_polys adalah quad 4 titik (teks miring).
    Keduanya direduksi ke persegi pembatas — cukup untuk highlight "sumber nilai".
    """
    try:
        pts = np.asarray(value, dtype=float).reshape(-1, 2)
    except (ValueError, TypeError):
        return None
    if pts.size == 0:
        return None
    return [int(pts[:, 0].min()), int(pts[:, 1].min()), int(pts[:, 0].max()), int(pts[:, 1].max())]


def _nonempty(value: Any) -> bool:
    # Jangan `value or ...`: array numpy melempar "truth value is ambiguous".
    return value is not None and len(value) > 0


def _rotation(inner: dict) -> int:
    """Sudut dari `doc_preprocessor_res.angle` (PaddleX: -1 bila klasifikasi orientasi mati)."""
    pre = inner.get("doc_preprocessor_res")
    if not isinstance(pre, dict):
        return 0
    try:
        angle = int(pre.get("angle", 0))
    except (TypeError, ValueError):
        return 0
    return angle if angle in _ROTATIONS else 0


def parse_predict_result(results: Any) -> EngineResult:
    """Ubah output `PaddleOCR.predict()` (3.x) menjadi EngineResult, urutan baris dipertahankan."""
    out = EngineResult()
    for res in results or []:
        data = res.json if hasattr(res, "json") else res
        if not isinstance(data, dict):
            continue
        inner = data.get("res", data)
        texts = inner.get("rec_texts")
        texts = list(texts) if _nonempty(texts) else []
        scores = inner.get("rec_scores")
        scores = list(scores) if _nonempty(scores) else []
        raw = inner.get("rec_boxes")
        if not _nonempty(raw):
            raw = inner.get("rec_polys")
        if not _nonempty(raw):
            raw = inner.get("dt_polys")
        boxes = [to_box(b) for b in raw] if _nonempty(raw) else []

        for i, text in enumerate(texts):
            score = scores[i] if i < len(scores) else None
            box = boxes[i] if i < len(boxes) else None
            if score is None or box is None:
                # Kontrak mewajibkan confidence & bbox. Tidak terjadi di 3.x normal; dibuang dan
                # dihitung agar `text` tetap = gabungan `lines`.
                out.dropped_lines += 1
                continue
            conf = min(max(float(score), 0.0), 1.0)
            out.lines.append(EngineLine(text=str(text), confidence=conf, bbox=box))
        # Satu gambar = satu hasil; bila ada beberapa, rotasi pertama yang bukan 0 dipakai.
        if out.rotation_applied == 0:
            out.rotation_applied = _rotation(inner)
    return out


# --- PaddleOCR ---------------------------------------------------------------------------------


def detect_device() -> str:
    import paddle

    # Wheel CPU dipasang untuk dev (tidak ada build CUDA di macOS), wheel GPU hanya di mesin NVIDIA,
    # jadi tanyakan ke wheel yang terpasang, jangan di-hardcode.
    if paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0:
        return "gpu"
    return "cpu"


class PaddleEngine:
    engine_id = ENGINE_ID

    def __init__(self, settings: Settings, device: str):
        import paddleocr
        from paddleocr import PaddleOCR

        self.version: str = str(getattr(paddleocr, "__version__", "unknown"))
        self._settings = settings
        # lang="en" cukup untuk dokumen Latin (KTP, PKKPR, ...). Tinjau ulang bila ada salah baca
        # sistematis pada karakter khas Indonesia. Flag pra-proses hanya dikirim bila di-set; tanpa itu
        # PaddleOCR memakai default-nya, sama seperti existing (item 24).
        flags = {
            name: value
            for name in ("use_doc_orientation_classify", "use_doc_unwarping", "use_textline_orientation")
            if (value := getattr(settings, name)) is not None
        }
        self._ocr = PaddleOCR(lang=settings.lang, device=device, enable_mkldnn=settings.enable_mkldnn, **flags)

    def predict(self, image: Image.Image) -> EngineResult:
        result = parse_predict_result(self._ocr.predict(np.asarray(image.convert("RGB"))))
        if self._settings.use_doc_orientation_classify is False:
            result.rotation_applied = 0
        return result


class PaddleEngineFactory:
    def __init__(self, settings: Settings):
        self._settings = settings
        self._device: str | None = None

    def device(self) -> str:
        if self._device is None:
            self._device = detect_device() if self._settings.device == "auto" else self._settings.device
        return self._device

    def create(self) -> Engine:
        return PaddleEngine(self._settings, self.device())
