<h1 align="center">Generative Reasoning Recommendation via LLMs</h1>

This repository contains the official implementation for the paper "Generative Reasoning Recommendation via LLMs".

## Introduction

Despite their remarkable reasoning capabilities across diverse domains,
large language models (LLMs) face fundamental challenges in natively
functioning as **g**enerative **r**easoning **r**ecommendation
**m**odels (**GRRMs**), where the intrinsic modeling gap between textual
semantics and collaborative filtering signals, combined with the
sparsity and stochasticity of user feedback, presents significant
obstacles. This work explores how to build GRRMs by adapting pre-trained
LLMs, which achieves a unified *understanding-reasoning-prediction*
manner for recommendation tasks. Towards this, we propose , an
end-to-end framework that integrates three components: (i)
**Collaborative-Semantic Alignment**, which fuses heterogeneous textual
evidence to construct semantically consistent, discrete item indices and
auxiliary alignment tasks that ground linguistic representations in
interaction semantics; (ii) **Reasoning Curriculum Activation**, which
builds a synthetic dataset with explicit Chain-of-Thought supervision
and a curriculum that progresses through behavioral evidence extraction,
latent preference modeling, intent inference, recommendation
formulation, and denoised sequence rewriting; and (iii)
**Sparse-Regularized Group Policy Optimization (SRPO)**, which
stabilizes post-training via *Residual-Sensitive Verifiable Reward* and
*Bonus-Calibrated Group Advantage Estimation*, enabling end-to-end
optimization under verifiable signals despite sparse successes. Distinct
from prior LLM recommenders that trade off efficiency for
interpretability, natively supports two complementary inference modes:
*Direct Sequence Recommendation* for high-throughput, low-latency
deployment, and *Sequential Reasoning Recommendation* that first emits
an interpretable reasoning chain for causal transparency. Extensive
experiments on three public benchmarks show consistent gains over strong
generative and LLM baselines in both direct and reasoning settings. The
results demonstrate that achieves a balanced trade-off among efficiency,
accuracy, and end-to-end interpretability, providing a practical path
for verifiable-RL-driven LLM recommenders.

<img src="assets/framework.png" alt="framework">

## Installation

Set up your environment with the required packages.

   ```bash
   bash scripts/install.sh
   ```

## Data Preparation

You can download data from [here](https://huggingface.co/datasets/Frywind/POLM_data). Put `data.zip` under this directory and put `sft_data.zip` under `LLaMA-Factory/data/`. Then unzip them.

   You can refer to `data_processing/` for instructions on how to prepare your dataset.

## Training

For sft training, we use `LLaMA-Factory`. Please refer to their [repository](./LLaMA-Factory) for more details. 

   You need to run `scripts/construct_model.py` to get Qwen3-4B-Instruct with extended vocabulary before sft training.

For rl training, update the configuration in `scripts/run.sh`, then run:

```bash
bash scripts/run.sh
```

## Evaluation

To evaluate the model on Amazon datasets, run:

   ```bash
   # For direct evaluation
   torchrun --nproc_per_node=8 --master_port=23324 eval/test_ddp_direct.py \
      --ckpt_path [CKPT_PATH] \
      --dataset [DATASET_NAME] \
      --results_file [RESULTS_JSON_FILE] \
      --test_batch_size 8 \
      --num_beams 10 \
      --index_file .index.json \
      --test_task seqrec \
      --test_prompt_ids 5

   # For reason evaluation, you need to deploy sglang servers first.

   # You can use our deployment script. Base port is 10010.
   bash scripts/deploy.sh [MODEL_PATH] [SERVE_NAME] [CUDA_DEVICES]

   torchrun --nproc_per_node=8 --master_port=23324 eval/test_ddp_reason.py \
      --ckpt_path [CKPT_PATH] \
      --vllm_model_name [SERVE_NAME] \
      --dataset [DATASET_NAME] \
      --results_file [RESULTS_JSONL_FILE] \
      --test_batch_size 4 \
      --num_beams 10 \
      --index_file .index.json \
      --test_task seqrec-rl \
      --test_prompt_ids 5
   ```