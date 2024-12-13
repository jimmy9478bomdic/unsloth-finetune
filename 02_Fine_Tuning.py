# Step 6: Fine-Tuning the Model
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq, Trainer
from unsloth import is_bfloat16_supported
from datasets import load_dataset
import torch

class AdaptiveTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_eval_loss = float('inf')

    def evaluation_step(self, *args, **kwargs):
        output = super().evaluation_step(*args, **kwargs)
        current_eval_loss = output['eval_loss']

        # Adaptive Learning Rate Adjustment
        if current_eval_loss > self.prev_eval_loss:
            self.args.learning_rate *= 0.9  # Reduce learning rate if loss increased
            print(f"Decreased learning rate to: {self.args.learning_rate}")
        else:
            self.args.learning_rate *= 1.05  # Slightly increase if loss decreased
            print(f"Increased learning rate to: {self.args.learning_rate}")

        self.prev_eval_loss = current_eval_loss
        return output

    def training_step(self, *args, **kwargs):
        # Adjust gradient clipping based on gradient norms
        if self.state.global_step > 0 and self.state.global_step % self.args.eval_steps == 0:
            current_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            print(f"Adjusted gradient clipping to: {current_grad_norm}")

        return super().training_step(*args, **kwargs)

def print_memory_stats(stage):
    gpu_stats = torch.cuda.get_device_properties(0)
    used_memory = round(torch.cuda.memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"[{stage}] GPU: {gpu_stats.name}, Memory Reserved: {used_memory} GB / {max_memory} GB")

max_seq_length = 2048
dtype = None
load_in_4bit = True

print("Loading model")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-2-9b",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token="token"
)

print("Loading Laura")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

print("Loading dataset")
dataset_path = "test_dataset.json"
dataset = load_dataset("json", data_files=dataset_path, split="train")

custom_prompt = """Source: {}
File: {}
Label: {}
Content: {}
"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    sources = examples["source"]
    files = examples["file"]
    labels = examples["label"]
    contents = examples["content"]
    texts = []
    for source,  file, label, content in zip(sources, files, labels, contents):
        text = custom_prompt.format(source, file, label, content) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        max_steps = 60,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

print_memory_stats("Before Training")
# 如果你需要從頭開始訓練，請將 resume_from_checkpoint 設置為 False
trainer_stats = trainer.train(resume_from_checkpoint=False)
print_memory_stats("After Training")

model.save_pretrained("lora_model")
tokenizer.save_pretrained("lora_model")