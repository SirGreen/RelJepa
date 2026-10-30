# Exact TVE environment.
# Python 3.10.14, PyTorch 2.1.2 (CUDA 12.1), PyG 2.4.0, pyg_lib 0.3.1, PyTorch Frame 0.2.3,
# RelBench 1.1.0, plus TVE's requirements.txt pins. The CUDA runtime ships inside the PyTorch
# wheel; the host driver is injected by Docker Desktop (--gpus all).
FROM python:3.10.14-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends git procps \
    && rm -rf /var/lib/apt/lists/*

# The base image's pip 23.0 rejects current typing_extensions wheels (name normalisation) and
# falls back to an sdist build that fails on the PyTorch-only index, so upgrade pip and keep PyPI.
RUN pip install --upgrade "pip==24.2" \
    && pip install torch==2.1.2+cu121 \
        --index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.org/simple

RUN pip install torch_geometric==2.4.0 \
    && pip install pyg_lib==0.3.1 torch_scatter==2.1.2 torch_sparse==0.6.18 \
        torch_cluster==1.6.3 torch_spline_conv==1.2.2 \
        -f https://data.pyg.org/whl/torch-2.1.2+cu121.html

RUN pip install pytorch_frame==0.2.3 relbench==1.1.0

# TVE's own pins (third_party/task-vector-estimation/requirements.txt), installed last so they win.
COPY requirements-tve.txt /tmp/requirements-tve.txt
RUN pip install -r /tmp/requirements-tve.txt

# Not listed by TVE but imported (prince: MCA in task_vector_generation_helper) or needed by us.
COPY requirements-extra.txt /tmp/requirements-extra.txt
RUN pip install -r /tmp/requirements-extra.txt

WORKDIR /workspace
