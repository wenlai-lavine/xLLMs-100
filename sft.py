# Fine-Tune Llama2-7b on SE paired dataset
import os
from dataclasses import dataclass, field
from typing import Optional

import torch
import tyro
from accelerate import Accelerator
from datasets import load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments

from trl import SFTTrainer
from trl.import_utils import is_xpu_available
from trl.trainer import ConstantLengthDataset

# define the prompt dict, include with input and without input
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}

@dataclass
class ScriptArguments:
    model_name: Optional[str] = field(default="meta-llama/Llama-2-7b-chat-hf", metadata={"help": "the model name"})
    data_path: Optional[str] = field(default="/fs/scratch/rng_cr_bcai_dl/law1rng/lavine_bosch/lavine_code/RLCLF/data_preprocess/multi_alpaca/training_data/total_multi_alpaca.json", metadata={"help": "the dataset name"})
    subset: Optional[str] = field(default="data/finetune", metadata={"help": "the subset to use"})
    split: Optional[str] = field(default="train", metadata={"help": "the split to use"})
    size_valid_set: Optional[int] = field(default=4000, metadata={"help": "the size of the validation set"})
    streaming: Optional[bool] = field(default=True, metadata={"help": "whether to stream the dataset"})
    shuffle_buffer: Optional[int] = field(default=5000, metadata={"help": "the shuffle buffer size"})
    seq_length: Optional[int] = field(default=1024, metadata={"help": "the sequence length"})
    num_workers: Optional[int] = field(default=4, metadata={"help": "the number of workers"})
    # add new
    cache_dir: Optional[str] = field(default="/fs/scratch/rng_cr_bcai_dl/law1rng/.cache", metadata={"help": "the model name"})

    training_args: TrainingArguments = field(
        default_factory=lambda: TrainingArguments(
            output_dir="./results",
            max_steps=300000,
            # num_train_epochs=1,
            logging_steps=10,
            save_strategy="steps",
            save_steps=2000,
            evaluation_strategy="steps",
            eval_steps=2000,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            gradient_accumulation_steps=8,
            gradient_checkpointing=False,
            group_by_length=False,
            learning_rate=1e-4,
            lr_scheduler_type="cosine",
            warmup_steps=100,
            weight_decay=0.05,
            optim="paged_adamw_32bit",
            bf16=True,
            remove_unused_columns=False,
            run_name="sft_multi_alpaca_llama2",
            report_to="wandb",
            logging_dir="./sft_multi_alpaca_logging",
            hub_token="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"
        )
    )

    packing: Optional[bool] = field(default=True, metadata={"help": "whether to use packing for SFTTrainer"})

    peft_config: LoraConfig = field(
        default_factory=lambda: LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
    )


script_args = tyro.cli(ScriptArguments)

if script_args.training_args.group_by_length and script_args.packing:
    raise ValueError("Cannot use both packing and group by length")

# `gradient_checkpointing` was True by default until `1f3314`, but it's actually not used.
# `gradient_checkpointing=True` will cause `Variable._execution_engine.run_backward`.
if script_args.training_args.gradient_checkpointing:
    raise ValueError("gradient_checkpointing not supported")


def chars_token_ratio(dataset, tokenizer, nb_examples=400):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(example):
    """Prepare the text from a sample of the dataset."""
    if example['input']:
        prompt_input = PROMPT_DICT['prompt_input'].format(
        instruction=example["instruction"],
        input=example["input"]
    )
    else:
        prompt_input = PROMPT_DICT['prompt_no_input'].format(
            instruction=example['instruction']
        )

    prompt_res = f"Question: {prompt_input}\n\nAnswer: {example['output']}"
    return prompt_res


def create_datasets(tokenizer, args):
    dataset = load_dataset(
        'json',
        split=args.split,
        data_files=args.data_path,
        use_auth_token=True,
        num_proc=args.num_workers if not args.streaming else None,
        streaming=args.streaming,
    )
    if args.streaming:
        print("Loading the dataset in streaming mode")
        valid_data = dataset.take(args.size_valid_set)
        train_data = dataset.skip(args.size_valid_set)
        train_data = train_data.shuffle(buffer_size=args.shuffle_buffer, seed=None)
    else:
        dataset = dataset.train_test_split(test_size=0.005, seed=None)
        train_data = dataset["train"]
        valid_data = dataset["test"]
        print(f"Size of the train set: {len(train_data)}. Size of the validation set: {len(valid_data)}")

    chars_per_token = chars_token_ratio(train_data, tokenizer)
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    train_dataset = ConstantLengthDataset(
        tokenizer,
        train_data,
        formatting_func=prepare_sample_text,
        infinite=True,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer,
        valid_data,
        formatting_func=prepare_sample_text,
        infinite=False,
        seq_length=args.seq_length,
        chars_per_token=chars_per_token,
    )
    return train_dataset, valid_dataset


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    script_args.model_name,
    quantization_config=bnb_config,
    device_map={"": Accelerator().local_process_index},
    trust_remote_code=True,
    use_auth_token=True,
    cache_dir=script_args.cache_dir,
)
base_model.config.use_cache = False

peft_config = script_args.peft_config

tokenizer = AutoTokenizer.from_pretrained(script_args.model_name, trust_remote_code=True, cache_dir=script_args.cache_dir)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

training_args = script_args.training_args

train_dataset, eval_dataset = create_datasets(tokenizer, script_args)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config,
    packing=script_args.packing,
    max_seq_length=None,
    tokenizer=tokenizer,
    args=training_args,
)
trainer.train()
trainer.save_model(script_args.training_args.output_dir)

output_dir = os.path.join(script_args.training_args.output_dir, "final_checkpoint")
trainer.model.save_pretrained(output_dir)

# Free memory for merging weights
del base_model
if is_xpu_available():
    torch.xpu.empty_cache()
else:
    torch.cuda.empty_cache()

model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

output_merged_dir = os.path.join(script_args.training_args.output_dir, "final_merged_checkpoint")
model.save_pretrained(output_merged_dir, safe_serialization=True)
