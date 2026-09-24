"""Generate `contract/` dari `ocr_service.contract`.

    python -m ocr_service.export_contract            # tulis ulang
    python -m ocr_service.export_contract --check    # exit 1 bila file basi (dipakai juga di test)

Contoh berisi data sintetis saja (tidak ada dokumen asli).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ocr_service.contract import ErrorResponse, OcrMetadata, OcrResponse

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "contract"

_REQUEST_ID = "job_0000example:page-2"

EXAMPLE_METADATA = OcrMetadata(request_id=_REQUEST_ID)

EXAMPLE_RESPONSE = OcrResponse(
    request_id=_REQUEST_ID,
    engine="paddleocr",
    engine_version="3.7.0",
    text="SERTIFIKAT HAK GUNA BANGUNAN\nNomor: 00.00.00.00.0.00000",
    lines=[
        {"text": "SERTIFIKAT HAK GUNA BANGUNAN", "confidence": 0.97, "bbox": [112, 80, 890, 132]},
        {"text": "Nomor: 00.00.00.00.0.00000", "confidence": 0.58, "bbox": [112, 150, 540, 188]},
    ],
    rotation_applied=0,
    timing_ms=840,
)

EXAMPLE_RESPONSE_EMPTY = EXAMPLE_RESPONSE.model_copy(update={"text": "", "lines": [], "timing_ms": 310})

EXAMPLE_ERROR_503 = ErrorResponse(request_id=_REQUEST_ID, error_code="busy", message="semua engine sedang dipakai")
EXAMPLE_ERROR_415 = ErrorResponse(
    request_id=_REQUEST_ID, error_code="unsupported_media_type", message="image bukan PNG"
)


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False) + "\n"


def render() -> dict[str, str]:
    """Nama file → isi. Satu-satunya definisi isi `contract/`."""
    return {
        "ocr-v1.metadata.schema.json": _dump(OcrMetadata.model_json_schema()),
        "ocr-v1.response.schema.json": _dump(OcrResponse.model_json_schema()),
        "ocr-v1.error.schema.json": _dump(ErrorResponse.model_json_schema()),
        "example_request_metadata.json": _dump(EXAMPLE_METADATA.model_dump()),
        "example_response.json": _dump(EXAMPLE_RESPONSE.model_dump()),
        "example_response_empty.json": _dump(EXAMPLE_RESPONSE_EMPTY.model_dump()),
        "example_error_503.json": _dump(EXAMPLE_ERROR_503.model_dump()),
        "example_error_415.json": _dump(EXAMPLE_ERROR_415.model_dump()),
    }


def stale_files(directory: Path = DEFAULT_DIR) -> list[str]:
    stale = []
    for name, content in render().items():
        path = directory / name
        if not path.exists() or path.read_text(encoding="utf-8") != content:
            stale.append(name)
    return stale


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--check", action="store_true", help="jangan tulis; exit 1 bila basi")
    args = parser.parse_args(argv)
    if args.check:
        stale = stale_files(args.out)
        for name in stale:
            print(f"basi: {args.out / name}", file=sys.stderr)
        return 1 if stale else 0
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in render().items():
        (args.out / name).write_text(content, encoding="utf-8")
        print(args.out / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
