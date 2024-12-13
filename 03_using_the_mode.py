from unsloth import FastLanguageModel
from transformers import TextStreamer
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = True

print("Loading fine-tuned model")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
# alpaca_prompt = You MUST copy from above!

inputs = tokenizer(
    [
        """
        我覺得很疲勞，可以吃什麼改善疲勞?
        """
    ], return_tensors = "pt").to("cuda")

outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
tokenizer.batch_decode(outputs)
print(tokenizer.batch_decode(outputs))