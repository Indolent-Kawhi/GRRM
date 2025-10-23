## Data Preparation

1. Get Amazon Dataset and store it like below:

   ```
   data
    └── Amazon
         ├── Ratings
         │   ├── ...
         │   └── ratings_Sports_and_Outdoors.csv
         ├── Review
         │   ├── ...
         │   └── reviews_Sports_and_Outdoors_5.json.gz
         └── Metadata
             ├── ...
             └── meta_Sports_and_Outdoors.json.gz
   ```

2. Process Amazon Dataset by running:

   ```bash
   # For Amazon-14
   python data_process/preprocess/amazon14_data_process.py --dataset [DATASET_NAME] --input data/Amazon --output data
    # For Amazon-18
   python data_process/preprocess/amazon18_data_process.py --dataset [DATASET_NAME] --input data/Amazon --output data
   ```

3. Synthesize preference data thorough:

   ```bash
   # If you use remote API
   python data_process/preprocess/construct_preference.py --dataset [DATASET_NAME] --api_key [YOUR_API_KEY] --base_url [BASE_URL] --model_name [MODEL_NAME]
   # If you use local deployment
   python data_process/preprocess/construct_preference.py --dataset [DATASET_NAME] --ports [PORTS] --model_name [MODEL_NAME]
   ```

4. Index the generated data by running:

   ```bash
   torchrun --nproc_per_node=8 data_process/preprocess/text_emb.py --dataset [DATASET_NAME] --plm_checkpoint [MODEL_PATH]
   python data_process/preprocess/rkmeans_constrained.py --dataset [DATASET_NAME]
   ```

5. Synthesize reason data:

   ```bash
   # If you use remote API
   python data_process/construct_cot.py --dataset [DATASET_NAME] --api_key [YOUR_API_KEY] --base_url [BASE_URL] --model_name [MODEL_NAME]
   # If you use local deployment
   python data_process/construct_cot.py --dataset [DATASET_NAME] --ports [PORTS] --model_name [MODEL_NAME]
   ```

5. Get sft data by running:

   ```bash
   python data_process/get_align.py \
      --dataset [DATASET] \
      --epochs 3 \
      --tasks seqrec,item2index,index2item,fusionseqrec,itemsearch,preferenceobtain \
      --train_prompt_sample_num 1,2,2,1,1,1 \
      --train_data_sample_num 0,0,0,50000,0,0 \
      --index_file .index.json

   python data_process/transform.py --dataset [DATASET_NAME]
   ```

6. Get rl data by running:

   ```bash
   python data_process/transform_rl.py --dataset [DATASET_NAME]
   ```