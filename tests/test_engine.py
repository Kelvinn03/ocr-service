"""Parser hasil PaddleOCR 3.x (port test existing `tests/test_ocr.py`) + PaddleEngine nyata bila ada."""

import importlib.util

import numpy as np
import pytest

from ocr_service.engine import parse_predict_result, to_box


def _lines(res):
    return [(ln.text, ln.confidence, ln.bbox) for ln in res.lines]


def test_to_box_reduces_a_quad_to_a_bounding_rectangle():
    assert to_box([[10, 20], [50, 18], [52, 40], [12, 42]]) == [10, 18, 52, 42]
    assert to_box([10, 20, 50, 40]) == [10, 20, 50, 40]
    assert to_box(np.array([10.7, 20.2, 50.9, 40.1])) == [10, 20, 50, 40]


@pytest.mark.parametrize("bad", [[], None, "nonsense", [[1]], {}])
def test_to_box_returns_none_for_unusable_input(bad):
    assert to_box(bad) is None


def test_reads_the_3x_shape_in_engine_order():
    res = parse_predict_result(
        [{"res": {"rec_texts": ["B", "A"], "rec_scores": [0.98, 0.71], "rec_boxes": [[0, 0, 10, 10], [0, 12, 20, 22]]}}]
    )
    assert _lines(res) == [("B", 0.98, [0, 0, 10, 10]), ("A", 0.71, [0, 12, 20, 22])]
    assert res.rotation_applied == 0 and res.dropped_lines == 0


def test_numpy_arrays_do_not_trip_truthiness():
    res = parse_predict_result(
        [{"res": {
            "rec_texts": ["A"], "rec_scores": np.array([np.float32(0.5)]),
            "rec_boxes": np.zeros((0, 4)), "rec_polys": np.array([[[1, 2], [9, 2], [9, 8], [1, 8]]]),
        }}]
    )  # fmt: skip
    assert _lines(res) == [("A", 0.5, [1, 2, 9, 8])]
    assert isinstance(res.lines[0].confidence, float)


def test_falls_back_from_boxes_to_polygons():
    res = parse_predict_result([{"res": {"rec_texts": ["A"], "rec_scores": [0.9], "dt_polys": [[[3, 4], [7, 4], [7, 6], [3, 6]]]}}])
    assert res.lines[0].bbox == [3, 4, 7, 6]


def test_result_object_with_json_attribute():
    class _Res:
        json = {"res": {"rec_texts": ["X"], "rec_scores": [0.5], "rec_boxes": [[0, 0, 1, 1]]}}

    assert _lines(parse_predict_result([_Res()])) == [("X", 0.5, [0, 0, 1, 1])]


@pytest.mark.parametrize("results", [[], None, [{"res": {}}], [{"res": {"rec_texts": None, "rec_scores": None}}]])
def test_tolerates_missing_and_null_keys(results):
    res = parse_predict_result(results)
    assert res.lines == [] and res.rotation_applied == 0


def test_lines_without_score_or_box_are_dropped_and_counted():
    res = parse_predict_result([{"res": {"rec_texts": ["A", "B"], "rec_scores": [0.9], "rec_boxes": [[0, 0, 1, 1]]}}])
    assert [ln.text for ln in res.lines] == ["A"] and res.dropped_lines == 1


def test_confidence_is_clamped_to_unit_interval():
    res = parse_predict_result([{"res": {"rec_texts": ["A"], "rec_scores": [1.0000001], "rec_boxes": [[0, 0, 1, 1]]}}])
    assert res.lines[0].confidence == 1.0


@pytest.mark.parametrize(("pre", "expected"), [({"angle": 90}, 90), ({"angle": -1}, 0), ({"angle": 45}, 0), (None, 0)])
def test_rotation_from_doc_preprocessor(pre, expected):
    inner = {"rec_texts": [], "rec_scores": [], "rec_boxes": []}
    if pre is not None:
        inner["doc_preprocessor_res"] = pre
    assert parse_predict_result([{"res": inner}]).rotation_applied == expected


@pytest.mark.paddle
@pytest.mark.skipif(importlib.util.find_spec("paddleocr") is None, reason="paddleocr tidak terpasang")
def test_real_paddle_engine_on_synthetic_png():  # pragma: no cover - hanya di mesin dengan paddle
    from PIL import Image, ImageDraw

    from ocr_service.engine import PaddleEngineFactory
    from ocr_service.settings import Settings

    img = Image.new("RGB", (600, 120), "white")
    ImageDraw.Draw(img).text((20, 40), "HELLO OCR 12345", fill="black")
    factory = PaddleEngineFactory(Settings(_env_file=None, device="cpu"))
    res = factory.create().predict(img)
    assert all(0.0 <= ln.confidence <= 1.0 and len(ln.bbox) == 4 for ln in res.lines)
