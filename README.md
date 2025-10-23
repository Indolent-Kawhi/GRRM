1. Set up your environment with the required packages.

   ```bash
   bash scripts/install.sh
   ```

2. You can download data from [here](https://huggingface.co/datasets/Frywind/POLM_data). Put `data.zip` under this directory and put `sft_data.zip` under `LLaMA-Factory/data/`. Then unzip them.

   You can refer to `data_processing/` for instructions on how to prepare your dataset.


3. For sft training, we use `LLaMA-Factory`. Please refer to their [repository](./LLaMA-Factory) for more details. 

   You need to run `scripts/construct_model.py` to get Qwen3-4B-Instruct with extended vocabulary before sft training.

4. For rl training, update the configuration in `scripts/run.sh`, then run:

   ```bash
   bash scripts/run.sh
   ```

5. To evaluate the model on Amazon datasets, run:

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