from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

def formatting_prompts_func(examples):
 convos = examples["conversations"]
 texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
 return { "text": texts }


max_seq_length = 2048
dtype = None # Auto detection of dtype
load_in_4bit = True # Reduce memory usage with 4-bit quantization
model, tokenizer = FastLanguageModel.from_pretrained(
 model_name = "unsloth/Llama-3.2-3B-Instruct",
 max_seq_length = max_seq_length,
 dtype = dtype,
 load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
 model,
 r = 16, # LoRA rank
 target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
 lora_alpha = 16,
 lora_dropout = 0, # Optimized at 0
 bias = "none", # No additional bias terms
 use_gradient_checkpointing = "unsloth", # Gradient checkpointing to save memory
 random_state = 3407,
 use_rslora = False, # Rank stabilized LoRA, can be enabled for stability
)

dataset = load_dataset("charanhu/kannada-instruct-dataset-390-k", split="train")
from unsloth.chat_templates import standardize_sharegpt
dataset = standardize_sharegpt(dataset)

dataset = dataset.map(formatting_prompts_func, batched=True)

trainer = SFTTrainer(
 model = model,
 tokenizer = tokenizer,
 train_dataset = dataset,
 dataset_text_field = "text",
 max_seq_length = max_seq_length,
 data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
 dataset_num_proc = 2,
 packing = False,
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
 ),
)

from unsloth.chat_templates import train_on_responses_only
trainer = train_on_responses_only(
 trainer,
 instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
 response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)
trainer_stats = trainer.train()

messages = [
 {"role": "user", "content": "1000"}
]
inputs = tokenizer.apply_chat_template(
 messages,
 tokenize=True,
 add_generation_prompt=True,
 return_tensors="pt",
).to("cuda")
outputs = model.generate(input_ids=inputs, max_new_tokens=1024, use_cache=True, temperature=1.5, min_p=0.1)
tokenizer.batch_decode(outputs)

model.save_pretrained("lora_test")
tokenizer.save_pretrained("lora_test")