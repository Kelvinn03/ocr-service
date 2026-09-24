"""ocr-service: PaddleOCR di balik `POST /v1/ocr`. Stateless, tidak mengenal domain perizinan."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ocr-service")
except PackageNotFoundError:  # dijalankan dari source tanpa install
    __version__ = "0.0.0+unknown"
