#!/bin/bash

pip install --no-cache-dir "vllm==0.8.5.post1" "torch==2.6.0" "torchvision==0.21.0" "torchaudio==2.6.0" "tensordict==0.6.2" torchdata

pip install "transformers[hf_xet]>=4.51.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow>=15.0.0" pandas \
    "ray[default]" codetiming hydra-core pylatexenc qwen-vl-utils wandb dill pybind11 liger-kernel mathruler \
    pytest py-spy pyext pre-commit ruff polars k_means_constrained

pip install "nvidia-ml-py>=12.560.30" "fastapi[standard]>=0.115.0" "optree>=0.13.0" "pydantic>=2.9" "grpcio>=1.62.1"

wget -nv https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && \
    pip install --no-cache-dir flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

wget -nv https://github.com/flashinfer-ai/flashinfer/releases/download/v0.2.5/flashinfer_python-0.2.5+cu124torch2.6-cp38-abi3-linux_x86_64.whl && \
    uv pip install --no-cache-dir flashinfer_python-0.2.5+cu124torch2.6-cp38-abi3-linux_x86_64.whl

pip intall -e .

cd LLaMA-Factory

pip install -e ".[torch,metrics,deepspeed]"
