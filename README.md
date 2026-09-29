# ocr-service

Service OCR bersama: menerima satu gambar halaman (PNG), menjalankan PaddleOCR di GPU, dan
mengembalikan teks, baris, confidence per baris, dan posisi (`bbox`).

Prinsip (guide `arsi_new.md`, "Stage OCR"):

- **Mengembalikan fakta; pemanggil yang menerapkan aturan.** Confidence per baris dikembalikan apa
  adanya. Statistik (mean, min, jumlah di bawah ambang) dan ambangnya adalah aturan M2.
- **Stateless.** Tidak menyimpan apa pun dan tidak mengenal jenis dokumen.
- **Tidak mencatat isi teks OCR maupun isi gambar ke log.** Log hanya berisi `request_id`, status,
  kode error, durasi, engine dan versinya (JSON satu baris per request).
- **Engine dikunci ke PaddleOCR** (item 15). DeepSeek/LightOn/Unlimited/Ollama dari repo existing
  tidak dibawa.

Repo sendiri, dipisah dari repo backend M2 (`verification-backend`) dengan riwayat foldernya. Tidak
meng-import kode M2 (dijaga `tests/test_isolation.py`); M2 hanya bergantung pada kontrak di
`contract/`, yang disalin (vendor) di repo backend.

## Kontrak `POST /v1/ocr`

Sumber kebenaran: `src/ocr_service/contract.py`. Turunannya di `contract/` (JSON Schema + contoh),
di-generate dengan:

```bash
.venv/bin/python -m ocr_service.export_contract           # tulis ulang
.venv/bin/python -m ocr_service.export_contract --check   # exit 1 bila basi (juga dicek test & CI)
```

Request `multipart/form-data`:

| Part | Isi |
|---|---|
| `image` | PNG satu halaman. Dicek dari magic bytes, bukan dari Content-Type part. |
| `metadata` | JSON `{"request_id": "..."}`. `request_id` wajib, pola `^[A-Za-z0-9._:\-]{1,200}$`, tanpa data pribadi. M2 mengirim `<job_id>:page-<n>`. |

Respons 200 (`contract/example_response.json`):

```json
{
  "request_id": "job_0000example:page-2",
  "engine": "paddleocr",
  "engine_version": "3.7.0",
  "text": "SERTIFIKAT HAK GUNA BANGUNAN\nNomor: 00.00.00.00.0.00000",
  "lines": [
    {"text": "SERTIFIKAT HAK GUNA BANGUNAN", "confidence": 0.97, "bbox": [112, 80, 890, 132]},
    {"text": "Nomor: 00.00.00.00.0.00000", "confidence": 0.58, "bbox": [112, 150, 540, 188]}
  ],
  "rotation_applied": 0,
  "timing_ms": 840
}
```

- `text` = `lines[].text` digabung dengan `"\n"`, sesuai urutan baca yang dikeluarkan PaddleOCR
  (tidak diurutkan ulang, sama seperti existing).
- `bbox` = `[x0, y0, x1, y1]` piksel integer. Polygon (teks miring) direduksi ke persegi pembatas.
- `confidence` = skor PaddleOCR mentah (tidak dibulatkan), dijepit ke [0, 1].
- `engine_version` = `paddleocr.__version__`.
- `rotation_applied` = rotasi halaman (0/90/180/270) dari doc orientation classifier; selalu 0 bila
  klasifikasi orientasi dimatikan (default).
- `timing_ms` = durasi inference di engine saja (tanpa waktu antre menunggu engine; waktu antre
  di-log sebagai `queue_ms`).
- **Hasil kosong** (`text: ""`, `lines: []`) adalah 200, bukan error. M2 menandainya `empty`.
- Baris tanpa skor atau posisi dari engine (tidak terjadi pada 3.x normal) dibuang dari `lines` dan
  `text`, dan jumlahnya di-log sebagai `dropped_lines`.

Error: body `{"request_id": ... | null, "error_code": ..., "message": ...}` (tanpa isi teks OCR).

| Status | `error_code` | Arti | Perlakuan di M2 |
|---|---|---|---|
| 400 | `bad_request` | part hilang, metadata bukan JSON, gambar kosong | permanen |
| 413 | `image_too_large` | melebihi `OCR_MAX_IMAGE_BYTES` / `OCR_MAX_IMAGE_PIXELS` | permanen |
| 415 | `unsupported_media_type` | bukan PNG | permanen |
| 422 | `invalid_metadata` / `invalid_image` | metadata tidak sesuai skema / PNG rusak | permanen |
| 503 | `busy` / `not_ready` | semua engine terpakai sampai `OCR_ACQUIRE_TIMEOUT_S` / pool belum siap; ada `Retry-After` | sementara |
| 500 | `engine_error` | engine melempar exception pada gambar yang valid | sementara (lihat bawah) |

500 sengaja bukan 4xx: gambar sudah lolos validasi, jadi kegagalan ada di engine (OOM GPU, bug
paddle) dan bisa hilang saat dicoba ulang. "Engine selalu gagal pada halaman yang sama" menjadi
permanen lewat batas percobaan di M2.

Versi: perubahan dalam `/v1` hanya aditif (field opsional baru). Perubahan yang merusak = `/v2`
dijalankan berdampingan selama migrasi pemanggil.

### Endpoint lain

- `GET /healthz`: proses hidup. 503 hanya bila pool **gagal** dibangun (liveness me-restart pod).
- `GET /readyz`: 200 setelah semua engine siap; berisi `device`, `pool_size`, `in_use`,
  `engine_version`. `device: "cpu"` di cluster berarti container jalan tanpa GPU (15.5x lebih lambat).

## Konfigurasi (env)

| Env | Default | Keterangan |
|---|---|---|
| `OCR_DEVICE` | `auto` | `auto` / `gpu` / `cpu`; auto bertanya ke wheel paddle |
| `OCR_ENGINE_POOL_SIZE` | `2` | ukuran pool di GPU (manifest: 6) |
| `OCR_CPU_POOL_SIZE` | `1` | ukuran pool di CPU |
| `OCR_ACQUIRE_TIMEOUT_S` | `30` | batas tunggu engine kosong sebelum 503 |
| `OCR_RETRY_AFTER_S` | `5` | nilai header `Retry-After` |
| `OCR_MAX_IMAGE_BYTES` | `52428800` | 50 MiB |
| `OCR_MAX_IMAGE_PIXELS` | `50000000` | A4 250 DPI ≈ 6 MP |
| `OCR_LANG` | `en` | harus sama dengan warm-up di Dockerfile |
| `OCR_ENABLE_MKLDNN` | `false` | oneDNN gagal di paddle 3.3.1; biarkan false |
| `OCR_USE_DOC_ORIENTATION_CLASSIFY` | tidak di-set | kosong = default PaddleOCR (sama dengan existing) |
| `OCR_USE_DOC_UNWARPING` | tidak di-set | kosong = default PaddleOCR (sama dengan existing) |
| `OCR_USE_TEXTLINE_ORIENTATION` | tidak di-set | kosong = default PaddleOCR (sama dengan existing) |
| `OCR_LOG_LEVEL` | `INFO` | |

## Perubahan dari server OCR existing (`ocr/service.py`)

- `POST /ocr` (body PNG mentah) → `POST /v1/ocr` multipart dengan `request_id`.
- Kunci baris `score`/`box`/`box_space` → `confidence`/`bbox`; confidence tidak dibulatkan.
- Statistik confidence (`avg_confidence`, `min_confidence`, `low_conf_line_count`, floor 0.7) serta
  `line_count`/`confidence_available` dihapus: dihitung M2 dari `lines`.
- Tambah `request_id`, `engine_version`, `rotation_applied`, `timing_ms`.
- Pool penuh: dulu menunggu tanpa batas, sekarang 503 + `Retry-After` setelah `OCR_ACQUIRE_TIMEOUT_S`.
- Input tidak valid: dulu 500, sekarang 400/413/415/422 (permanen).
- Pool dibangun saat startup (thread latar), bukan pada request pertama; `/readyz` baru.
- Fallback API PaddleOCR 2.x dihapus (versi dipin `paddleocr==3.7.0`); print diagnostik
  `[ocr-pool]` dihapus.
- **Flag pra-proses PaddleOCR mengikuti existing (item 24).** Existing tidak memasang flag
  orientasi/unwarping, sehingga memakai default PaddleOCR 3.x; di sini juga tidak dipasang kecuali
  env di-set. Konsekuensi yang dicatat: bila doc orientation/unwarping aktif, `bbox` mengacu ke gambar
  hasil pra-proses, bukan PNG yang dikirim, sehingga highlight traceback scan bisa bergeser. Perlu dicek
  di sampel asli sebelum diputuskan berbeda dari existing.

## Menjalankan

Lokal (tanpa GPU; macOS tidak punya wheel paddle GPU):

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
.venv/bin/pip install -e ".[dev]"            # tanpa paddle: cukup untuk test (FakeEngine)
.venv/bin/ruff check . && .venv/bin/pytest -q

# Menjalankan engine nyata di CPU (Linux x86, atau platform yang punya wheel paddlepaddle):
.venv/bin/pip install -e ".[dev,cpu]"
.venv/bin/uvicorn ocr_service.main:create_app --factory --port 8002 --workers 1
curl -s localhost:8002/readyz
curl -s -F image=@halaman.png -F 'metadata={"request_id":"dev:page-1"};type=application/json' \
  localhost:8002/v1/ocr
```

Tanpa paddle terpasang service tetap start, tetapi `/healthz` dan `/readyz` menjawab 503
(`pool_state: failed`).

Image GPU (context build = root repo; jalan sebagai uid 10001, cache model di `/home/ocr/.paddlex`).
`/opt/venv` dipasang di stage `ocr-venv` lalu di-COPY dalam 12 layer ≤ ~1 GiB (`docker/split_layers.py`):
satu layer ±3,7 GiB hasil install paddle GPU gagal di-push ke Harbor (500). Bila `split_layers` gagal
karena bucket tidak cukup, naikkan jumlahnya di baris `RUN` DAN tambah baris `COPY --from=ocr-venv`
(dijaga `tests/test_split_layers.py`).

```bash
docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t ocr-service .
docker run --gpus all -p 8002:8002 ocr-service
```

## Deploy

| Target | Workflow | Isi |
|---|---|---|
| Dev instance (docker compose milik repo backend, `~/m2`) | `deploy-dev.yml` | `ci` hijau di main → image ke GHCR (`:latest`, `:<sha>`) → SSH, tulis `OCR_IMAGE`/`OCR_TAG` ke `~/m2/.deploy-tags` → restart service `ocr` saja |
| Cluster AI (GitOps) | `build-push.yml` | `ci` hijau di main → image ke Harbor (`<HARBOR_HOST>/apps/ocr-service:<sha7>`) → ganti `newTag` di bkpm-gitops `apps/ai/ocr-service/` → ArgoCD sync |

Manifest Kubernetes **tidak** ada di repo ini, tetapi di repo bkpm-gitops (`apps/ai/ocr-service/`:
1 replika, `Recreate`, 1 GPU A10, pool 6, ClusterIP tanpa auth). Baca komentar di manifest itu
sebelum sync pertama, khususnya pengecekan GPU dan memory limit yang masih estimasi.

Secrets repo: `SSH_HOST`, `SSH_USERNAME`, `SSH_PRIVATE_KEY` (dev instance, sama dengan repo backend),
`HARBOR_USERNAME`, `HARBOR_PASSWORD`, `GITOPS_TOKEN` (dari admin platform). Variables: `HARBOR_HOST`,
`TARGET_CLUSTER=ai`.

Kontrak berubah → jalankan `export_contract`, rilis, lalu salin `contract/` ke repo backend
(`tests/fixtures/ocr_contract/`) agar test kompatibilitas client M2 ikut diperbarui.

## Yang harus diverifikasi di cluster GPU

- Dengan flag pra-proses default (sama dengan existing): cek apakah `bbox` masih sejalan dengan PNG
  yang dikirim pada scan miring/terbalik/melengkung (item 24). Pastikan juga `doc_preprocessor_res.angle`
  benar-benar yang dilaporkan PaddleOCR 3.7.0 saat orientasi dinyalakan.
- VRAM pool 6 (~16 GiB dari 23 GiB, estimasi; item 11/21) dan memory host (limit 8Gi = estimasi).
- `OCR_ACQUIRE_TIMEOUT_S` vs `OCR_PAGE_CONCURRENCY` M2 dan timeout client M2 (item 23/24).
- `engine_version` terbaca `3.7.0` dari image.
- Image kini jalan sebagai **non-root** (uid 10001, `HOME=/home/ocr`; existing: root). Cek di run
  pertama bahwa `/readyz` siap tanpa download model (cache terbaca dari `/home/ocr/.paddlex`).
