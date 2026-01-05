FROM nvcr.io/nvidia/pytorch:25.12-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# OS依存 + ビルド依存（flash-attnビルド用）
# ※ ninja は入れると並列ビルドになるが、DGX Sparkでは不安定報告もあるので入れない方針
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libgl1 libglib2.0-0 \
    build-essential \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install

WORKDIR /workspace/MemFlow

# 依存のキャッシュを効かせるため先にrequirementsだけ
COPY ../requirements.txt ./requirements.txt
COPY ../pyproject.toml ./pyproject.toml

# ---- requirements のうち Docker では除外するものをフィルタ ----
# - nvidia-tensorrt: PyPIはプレースホルダ（NVIDIA index専用）
# - nvidia-pyindex : インストール時にpip設定を書き換える実装で、Dockerビルド環境で壊れやすい
RUN python -V && pip --version \
 && pip install -U pip \
 && grep -vE '^(nvidia-tensorrt|nvidia-pyindex)([<=> ].*)?$' requirements.txt > /tmp/requirements.docker.txt \
 && pip install -r /tmp/requirements.docker.txt \
 && pip install "huggingface_hub[cli]" gradio

# ---- flash-attn: ソースからwheelを作って入れる（GB10 / sm121 対応のため）----
# GB10は compute capability 12.1 (sm_121) なのでアーキ指定
# flash-attentionビルドはメモリを大量消費（1ジョブあたり10-15GB）
# DGX Spark 128GBメモリでは MAX_JOBS=4 が最適
ARG FLASH_ATTN_VERSION="v2.8.3"
ENV TORCH_CUDA_ARCH_LIST="12.1" \
    MAX_JOBS=4 \
    NVCC_THREADS=2 \
    FLASH_ATTENTION_FORCE_BUILD=TRUE

RUN git clone --depth 1 --branch ${FLASH_ATTN_VERSION} https://github.com/Dao-AILab/flash-attention.git /tmp/flash-attn \
 && cd /tmp/flash-attn \
 && python setup.py bdist_wheel \
 && pip install dist/flash_attn-*.whl \
 && rm -rf /tmp/flash-attn

# リポジトリ本体をコピー（最後にやるとDockerレイヤキャッシュが効く）
COPY . .

EXPOSE 7860
ENTRYPOINT ["bash", "docker/entrypoint.sh"]

