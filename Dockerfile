# syntax=docker/dockerfile:1
#
# ocr-service: PaddleOCR di GPU di balik POST /v1/ocr. Port dari Dockerfile.ocr existing.
# Context build = folder ini saja (tidak ada file di luar ocr-service/).
#
#   docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) -t ocr-service .
#   docker run --gpus all -p 8002:8002 ocr-service
#
# Beda dari existing: hanya pyproject.toml, requirements-gpu.txt, docker/ dan src/ yang di-copy (existing `COPY . .` membawa seluruh
# repo karena config.py dan contract test satu pohon); image ini tidak butuh itu lagi.

# Harus cocok dengan driver NVIDIA host DAN index CUDA di requirements-gpu.txt (cu130). Varian cudnn
# wajib: model deteksi & rekognisi PaddleOCR berjalan lewat cuDNN, image -runtime biasa tidak membawanya.
ARG CUDA_IMAGE=nvidia/cuda:13.0.3-cudnn-runtime-ubuntu24.04


# =============================================================================
# Stage 1 -- bobot model PaddleOCR, di-fetch dengan wheel CPU.
#
# paddlepaddle-gpu tidak bisa di-IMPORT tanpa libcuda.so.1, yang disuntikkan NVIDIA Container Toolkit
# saat RUN, bukan bagian image; mesin build tidak punya GPU. Wheel CPU tidak me-link CUDA, jadi bisa
# di-import di sini, dan bobot yang di-download adalah file biasa yang sama (model inference; device
# hanya menentukan tempat jalan). Hanya direktori cache yang dibawa ke stage berikutnya.
# =============================================================================
FROM ${CUDA_IMAGE} AS ocr-models

ENV DEBIAN_FRONTEND=noninteractive \
    HOME=/root \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv libgl1 libglib2.0-0t64 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && pip install --no-cache-dir --upgrade pip

WORKDIR /warmup
COPY pyproject.toml ./
RUN mkdir -p src/ocr_service && touch src/ocr_service/__init__.py \
 && pip install --no-cache-dir ".[cpu]"

# lang= harus sama dengan Settings.lang (tests/test_dockerfile.py menjaga ini). Semua model opsional
# (doc orientation, UVDoc unwarping, textline orientation) ikut di-fetch walau default service
# mematikan dua yang pertama, agar menyalakannya lewat env tidak memicu download di dalam request.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(lang='en', enable_mkldnn=False, device='cpu', use_doc_orientation_classify=True, use_doc_unwarping=True, use_textline_orientation=True)"


# =============================================================================
# Stage 2 -- venv lengkap (paddle GPU + dependensi nvidia-*), lalu dipecah per ~1 GiB.
#
# Dipasang di stage terpisah dan di-COPY ke stage service dalam beberapa bagian (docker/split_layers.py)
# karena satu layer ±3,7 GiB gagal di-push ke Harbor (500), sedangkan layer <=1,4 GiB lolos. Isi
# /opt/venv di image akhir sama dengan install biasa.
# =============================================================================
FROM ${CUDA_IMAGE} AS ocr-venv

ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && pip install --no-cache-dir --upgrade pip

WORKDIR /build

# Dependensi dengan paket stub; kode asli dipasang di stage service.
COPY pyproject.toml ./
RUN mkdir -p src/ocr_service && touch src/ocr_service/__init__.py \
 && pip install --no-cache-dir ".[gpu]" \
 && pip uninstall -y ocr-service

# DUA pip run, bukan satu: --index-url di requirements-gpu.txt berlaku global untuk satu run, sehingga
# gabungan install akan mencari fastapi dkk. di index Paddle dan gagal.
COPY requirements-gpu.txt ./
RUN pip install --no-cache-dir -r requirements-gpu.txt

# Jumlah bucket harus sama dengan jumlah baris `COPY --from=ocr-venv` di bawah (dijaga tests/test_dockerfile.py).
COPY docker/split_layers.py ./
RUN python3 split_layers.py /opt/venv /split 12 1024


# =============================================================================
# Stage 3 -- service.
# =============================================================================
FROM ${CUDA_IMAGE} AS ocr

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Eksplisit dan load-bearing: PaddleOCR menyimpan cache bobot di $HOME/.paddlex. Warm-up di atas
    # mengisinya saat BUILD (di /root), lalu di-copy ke HOME user runtime; cache hit hanya bila HOME
    # menunjuk tempat cache itu.
    HOME=/home/ocr \
    PATH=/opt/venv/bin:$PATH

# libgomp1 = runtime OpenMP paddle. libgl1/libglib2.0-0t64 karena paddlex menarik opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 \
        python3-venv \
        libgl1 \
        libglib2.0-0t64 \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# User non-root (panduan deploy cluster). uid 10001 sama dengan image backend M2.
RUN groupadd --system --gid 10001 ocr \
 && useradd --system --uid 10001 --gid ocr --home-dir /home/ocr --create-home --shell /usr/sbin/nologin ocr

WORKDIR /srv/ocr

# /opt/venv dari stage ocr-venv, satu layer per bucket (masing-masing <= ~1 GiB belum terkompresi).
COPY --from=ocr-venv /split/0/ /
COPY --from=ocr-venv /split/1/ /
COPY --from=ocr-venv /split/2/ /
COPY --from=ocr-venv /split/3/ /
COPY --from=ocr-venv /split/4/ /
COPY --from=ocr-venv /split/5/ /
COPY --from=ocr-venv /split/6/ /
COPY --from=ocr-venv /split/7/ /
COPY --from=ocr-venv /split/8/ /
COPY --from=ocr-venv /split/9/ /
COPY --from=ocr-venv /split/10/ /
COPY --from=ocr-venv /split/11/ /

# ~177MB bobot model dari stage 1 (ocr-models). Tanpa ini download terjadi DI DALAM startup pod setiap deploy (dan
# gagal di host tanpa rute keluar). Sebelum kode agar layer ini bertahan di setiap perubahan kode.
# --chown: PaddleX boleh menulis ke cache-nya sendiri saat runtime (mis. file lock/metadata).
COPY --from=ocr-models --chown=10001:10001 /root/.paddlex /home/ocr/.paddlex

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps . && rm -rf build src/*.egg-info

ARG GIT_SHA=unknown
ARG BUILD_TIME=unknown
ENV BUILD_SHA=${GIT_SHA} \
    BUILD_TIME=${BUILD_TIME}

EXPOSE 8002

# /healthz = proses hidup (tidak membangun pool). /readyz = semua engine siap; field `device`
# memperlihatkan fallback CPU diam-diam bila container jalan tanpa --gpus.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD python3 -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8002/healthz', timeout=8).status == 200 else 1)"

# 0.0.0.0: dipanggil dari pod lain (ClusterIP). Satu worker: konkurensi ada di pool DI DALAM proses
# (engine berbagi satu CUDA context); worker kedua membayar CUDA context sendiri (~300-500MB).
# Skala dengan OCR_ENGINE_POOL_SIZE, bukan --workers.
USER 10001:10001

CMD ["uvicorn", "ocr_service.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8002", "--workers", "1"]
