"""`contract/` harus sama dengan hasil generate dari `ocr_service.contract`."""

import json

from ocr_service.contract import ErrorResponse, OcrMetadata, OcrResponse
from ocr_service.export_contract import DEFAULT_DIR, render, stale_files


def test_contract_files_are_not_stale():
    stale = stale_files(DEFAULT_DIR)
    assert not stale, f"contract/ basi: {stale}. Jalankan `python -m ocr_service.export_contract`."


def test_no_unexpected_files_in_contract_dir():
    assert {p.name for p in DEFAULT_DIR.glob("*.json")} == set(render())


def test_examples_validate_against_models():
    load = lambda name: json.loads((DEFAULT_DIR / name).read_text())  # noqa: E731
    OcrMetadata.model_validate(load("example_request_metadata.json"))
    for name in ("example_response.json", "example_response_empty.json"):
        body = OcrResponse.model_validate(load(name))
        assert body.text == "\n".join(ln.text for ln in body.lines)
    for name in ("example_error_503.json", "example_error_415.json"):
        ErrorResponse.model_validate(load(name))
