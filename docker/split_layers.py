"""Bagi isi satu direktori ke beberapa folder kecil agar setiap `COPY` menjadi layer image tersendiri.

Alasan: install `paddlepaddle-gpu` + dependensi `nvidia-*` menghasilkan SATU layer ±3,7 GiB (terkompresi),
dan push layer sebesar itu ke Harbor platform gagal 500 berulang, sementara layer ≤1,4 GiB lolos.
Isi image akhir tetap sama persis; hanya pembagian layernya yang berubah.

    python3 split_layers.py SRC DEST BUCKETS TARGET_MIB

File (dan symlink) di SRC dipindahkan ke DEST/<n>/<path absolut aslinya>, n = 0..BUCKETS-1, dengan
first-fit decreasing: file terbesar dulu, ke bucket pertama yang masih muat TARGET_MIB (ukuran belum
terkompresi). Satu file yang sendirian melebihi target tetap mendapat bucket sendiri. Semua direktori
dibuat ulang di bucket 0 supaya direktori kosong ikut terbawa. Semua bucket selalu dibuat (Dockerfile
meng-COPY masing-masing), dan skrip gagal bila BUCKETS tidak cukup.
"""

import os
import shutil
import sys


def plan(sizes: dict[str, int], buckets: int, target: int) -> dict[str, int]:
    """path -> nomor bucket. Deterministik: urut ukuran menurun, lalu path."""
    used = [0] * buckets
    out: dict[str, int] = {}
    for path, size in sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0])):
        for i in range(buckets):
            if used[i] == 0 or used[i] + size <= target:
                used[i] += size
                out[path] = i
                break
        else:
            raise SystemExit(f"split_layers: {buckets} bucket tidak cukup (target {target >> 20} MiB); naikkan BUCKETS")
    return out


def main(argv: list[str]) -> int:
    src, dest, buckets, target_mib = argv[1], argv[2], int(argv[3]), int(argv[4])
    src = os.path.abspath(src)
    sizes: dict[str, int] = {}
    dirs: list[str] = []
    for root, dirnames, filenames in os.walk(src):
        dirs.append(root)
        for name in dirnames:
            p = os.path.join(root, name)
            if os.path.islink(p):  # os.walk tidak masuk ke symlink direktori; pindahkan sebagai link
                sizes[p] = 0
        for name in filenames:
            p = os.path.join(root, name)
            sizes[p] = os.lstat(p).st_size
    assignment = plan(sizes, buckets, target_mib << 20)

    for i in range(buckets):
        os.makedirs(os.path.join(dest, str(i)), exist_ok=True)
    for d in dirs:
        target_dir = os.path.join(dest, "0", d.lstrip("/"))
        os.makedirs(target_dir, exist_ok=True)
        shutil.copystat(d, target_dir, follow_symlinks=False)
    for path, i in assignment.items():
        to = os.path.join(dest, str(i), path.lstrip("/"))
        os.makedirs(os.path.dirname(to), exist_ok=True)
        os.rename(path, to)

    totals = [0] * buckets
    for path, i in assignment.items():
        totals[i] += sizes[path]
    print("split_layers:", " ".join(f"{i}={t >> 20}MiB" for i, t in enumerate(totals)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
