import os
from transformers import AutoModelForCausalLM, AutoTokenizer

os.makedirs('models', exist_ok=True)

model_name = "Qwen/Qwen3-4B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
)

tokens_template = ["<|a_{}|>", "<|b_{}|>", "<|c_{}|>", "<|d_{}|>", "<|e_{}|>"]
tokens = []
for token in tokens_template:
    for i in range(1, 257):
        tokens.append(token.format(i))
num_added = tokenizer.add_special_tokens({
    "additional_special_tokens": tokens
})
model.resize_token_embeddings(len(tokenizer))

model.save_pretrained('models/qwen3-4b-instruct-5-256')
tokenizer.save_pretrained('models/qwen3-4b-instruct-5-256')