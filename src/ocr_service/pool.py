"""Pool engine PaddleOCR (port `ocr/pool.py` existing).

PaddleOCR tidak menjamin `predict()` thread-safe pada satu instance, jadi pool berisi N instance
independen dan tiap instance dipinjam satu pemanggil pada satu waktu. Di GPU N instance berbagi satu
CUDA context dan benar-benar berjalan bersamaan; skala lewat ukuran pool, bukan replika/worker.

Perbedaan dari existing:
- Dibangun saat startup (lifespan), bukan saat request pertama: alasan lazy existing ("web app
  mungkin tidak pernah OCR") tidak berlaku untuk service yang tugasnya hanya OCR. Tetap dibangun
  sekaligus, sehingga ukuran pool yang tidak muat di VRAM gagal keras di startup.
- `acquire(timeout)` → `PoolBusy` (dijawab 503 + Retry-After). Existing menunggu tanpa batas.
"""

import logging
import queue
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Literal

from ocr_service.engine import Engine, EngineFactory

log = logging.getLogger("ocr_service.pool")

PoolState = Literal["pending", "building", "ready", "failed"]


class PoolBusy(Exception):
    """Tidak ada engine kosong dalam batas acquire timeout."""


class PoolNotReady(Exception):
    """Pool belum selesai (atau gagal) dibangun."""


class EnginePool:
    def __init__(self, factory: EngineFactory, size_for: Callable[[str], int]):
        self._factory = factory
        self._size_for = size_for
        self._queue: queue.Queue[Engine] = queue.Queue()
        self._lock = threading.Lock()
        self._in_use = 0
        self.state: PoolState = "pending"
        self.device: str | None = None
        self.size = 0
        self.engine_version: str | None = None
        self.error: str | None = None

    def build(self) -> None:
        """Bangun semua engine sekaligus. Aman dipanggil dari thread latar."""
        with self._lock:
            if self.state != "pending":
                return
            self.state = "building"
        try:
            device = self._factory.device()
            size = self._size_for(device)
            engines = [self._factory.create() for _ in range(size)]
        except Exception as exc:
            # Hanya tipe exception: pesan dari paddle tidak berisi teks OCR, tetapi tidak perlu.
            self.error = type(exc).__name__
            self.state = "failed"
            log.exception("pool build gagal", extra={"event": "pool_build_failed"})
            return
        for engine in engines:
            self._queue.put(engine)
        self.device, self.size = device, size
        self.engine_version = engines[0].version if engines else None
        self.state = "ready"
        log.info(
            "pool siap",
            extra={"event": "pool_ready", "device": device, "pool_size": size},
        )

    @property
    def in_use(self) -> int:
        return self._in_use

    @contextmanager
    def acquire(self, timeout: float) -> Iterator[Engine]:
        """Pinjam satu engine; dikembalikan di `finally` agar error tidak membocorkan engine."""
        if self.state != "ready":
            raise PoolNotReady(self.state)
        try:
            engine = self._queue.get(timeout=timeout)
        except queue.Empty:
            raise PoolBusy from None
        with self._lock:
            self._in_use += 1
        try:
            yield engine
        finally:
            with self._lock:
                self._in_use -= 1
            self._queue.put(engine)
