"""docker/split_layers.py: pembagian /opt/venv ke beberapa layer image (lihat Dockerfile, stage ocr-venv)."""

import importlib.util
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("split_layers", ROOT / "docker" / "split_layers.py")
split_layers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(split_layers)

MiB = 1 << 20


def test_plan_keeps_buckets_under_target_and_is_deterministic():
    sizes = {"a": 854 * MiB, "b": 580 * MiB, "c": 462 * MiB, "d": 400 * MiB, "e": 10 * MiB, "f": 0}
    first = split_layers.plan(sizes, buckets=12, target=1024 * MiB)
    assert first == split_layers.plan(dict(reversed(list(sizes.items()))), buckets=12, target=1024 * MiB)
    totals: dict[int, int] = {}
    for path, bucket in first.items():
        totals[bucket] = totals.get(bucket, 0) + sizes[path]
    assert set(first) == set(sizes)
    assert all(t <= 1024 * MiB for t in totals.values())


def test_plan_gives_an_oversized_file_its_own_bucket():
    out = split_layers.plan({"huge": 2048 * MiB, "small": MiB}, buckets=2, target=1024 * MiB)
    assert out["huge"] != out["small"]


def test_plan_fails_loudly_when_buckets_run_out():
    with pytest.raises(SystemExit):
        split_layers.plan({"a": 900 * MiB, "b": 900 * MiB}, buckets=1, target=1024 * MiB)


def test_split_moves_everything_and_merging_buckets_restores_the_tree(tmp_path):
    src = tmp_path / "opt" / "venv"
    (src / "lib" / "pkg").mkdir(parents=True)
    (src / "empty").mkdir()
    (src / "bin").mkdir()
    (src / "lib" / "pkg" / "big.so").write_bytes(b"x" * 3000)
    (src / "lib" / "pkg" / "small.py").write_text("print(1)\n")
    os.chmod(src / "lib" / "pkg" / "big.so", 0o755)
    os.symlink("/usr/bin/python3", src / "bin" / "python")
    dest = tmp_path / "split"

    split_layers.main(["split_layers.py", str(src), str(dest), "3", "0"])  # target 0: satu file per bucket

    assert [p for p in src.rglob("*") if not p.is_dir()] == []  # semua file dipindah
    merged: dict[str, Path] = {}
    for bucket in sorted(dest.iterdir()):
        for p in bucket.rglob("*"):
            merged.setdefault(str(p.relative_to(bucket)), p)
    rel = str(src).lstrip("/")
    assert f"{rel}/empty" in merged  # direktori kosong ikut
    assert os.readlink(merged[f"{rel}/bin/python"]) == "/usr/bin/python3"
    assert merged[f"{rel}/lib/pkg/big.so"].read_bytes() == b"x" * 3000
    assert os.stat(merged[f"{rel}/lib/pkg/big.so"]).st_mode & 0o777 == 0o755


def test_dockerfile_copies_every_bucket_exactly_once():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    run = re.search(r"^RUN python3 split_layers\.py /opt/venv /split (\d+) (\d+)$", dockerfile, re.MULTILINE)
    assert run, "baris RUN split_layers.py tidak ditemukan"
    copies = re.findall(r"^COPY --from=ocr-venv /split/(\d+)/ /$", dockerfile, re.MULTILINE)
    assert copies == [str(i) for i in range(int(run.group(1)))]
