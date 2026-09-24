"""ocr-service harus bisa dipindah ke repo sendiri: tidak ada import dari backend `app`."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {"app"}


def _imported_roots(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_file_imports_backend_app():
    files = [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts]
    assert any(p.name == "api.py" for p in files)  # scan benar-benar menemukan source
    offenders = {str(p.relative_to(ROOT)): sorted(_imported_roots(p) & FORBIDDEN) for p in files}
    assert {k: v for k, v in offenders.items() if v} == {}


def test_no_paths_outside_the_folder_in_build_files():
    # Dockerfile/manifest tidak boleh merujuk file di luar ocr-service/.
    for name in ("Dockerfile", "pyproject.toml"):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "../" not in text, name
