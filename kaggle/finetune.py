"""Job B — QLoRA fine-tune on Kaggle T4.

Fine-tunes Qwen2.5-3B-Instruct on domain concept->explanation pairs
to produce sharper, more consistent explanations for CDP probing.
"""

import os
import json
import torch
from datetime import datetime, timezone

try:
    from kaggle_secrets import UserSecretsClient
    sec = UserSecretsClient()
    HF_TOKEN = sec.get_secret("HF_TOKEN")
    SB_URL = sec.get_secret("SUPABASE_URL")
    SB_KEY = sec.get_secret("SUPABASE_SERVICE_KEY")
except Exception:
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    SB_URL = os.environ.get("SUPABASE_URL", "")
    SB_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# Frozen hyperparams (from grid search on CV)
FROZEN_RANK = 16
FROZEN_LR = 2e-4
FROZEN_EPOCHS = 3


def main():
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer,
    )
    from peft import LoraConfig, get_peft_model
    from datasets import Dataset
    from supabase import create_client

    sb = create_client(SB_URL, SB_KEY)

    # Load training data: concept definitions from all domains
    concepts = sb.table("concepts").select(
        "canonical_name, definition"
    ).not_.is_("definition", "null").execute().data

    # Format as instruction pairs
    train_data = []
    for c in concepts:
        if c["definition"] and len(c["definition"]) > 20:
            train_data.append({
                "instruction": f"Explain the concept: {c['canonical_name']}",
                "output": c["definition"],
            })

    print(f"Training on {len(train_data)} concept-explanation pairs")
    dataset = Dataset.from_list(train_data)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, token=HF_TOKEN,
        load_in_4bit=True, device_map="auto",
    )

    # QLoRA config
    lora_cfg = LoraConfig(
        r=FROZEN_RANK,
        lora_alpha=FROZEN_RANK * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Tokenize
    def tokenize(example):
        text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        tokens = tokenizer(text, truncation=True, max_length=512, padding="max_length")
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Train
    args = TrainingArguments(
        output_dir="/kaggle/working/lora_output",
        num_train_epochs=FROZEN_EPOCHS,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=FROZEN_LR,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
    )
    trainer.train()

    # Save adapter
    adapter_path = "/kaggle/working/lora_adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")

    # Push to HF Hub
    if HF_TOKEN:
        repo_id = "lightgap/lightgap-cdp-adapter"
        model.push_to_hub(repo_id, token=HF_TOKEN, private=True)
        tokenizer.push_to_hub(repo_id, token=HF_TOKEN, private=True)
        print(f"Pushed to {repo_id}")


if __name__ == "__main__":
    main()
