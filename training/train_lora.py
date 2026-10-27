import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = os.getenv("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
DATA_FILE = os.getenv("PURPLE_TRAIN_FILE", "training/data.jsonl")
OUTPUT_DIR = os.getenv("PURPLE_TRAIN_OUTPUT", "purple-ai-lora")
MAX_LENGTH = int(os.getenv("PURPLE_TRAIN_MAX_LENGTH", "2048"))


def main():
    if not os.path.exists(DATA_FILE):
        raise SystemExit(
            f"Missing {DATA_FILE}. Copy training/data.example.jsonl to training/data.jsonl "
            "and replace examples with your own reviewed training data."
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="auto",
    )
    model.config.use_cache = False
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        ),
    )
    model.print_trainable_parameters()

    dataset = load_dataset("json", data_files=DATA_FILE, split="train")
    if "messages" not in dataset.column_names:
        raise SystemExit('Each JSONL row must contain a "messages" array with role/content objects.')

    def encode(row):
        text = tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return tokenizer(text, truncation=True, max_length=MAX_LENGTH)

    tokenized = dataset.map(encode, remove_columns=dataset.column_names)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=float(os.getenv("PURPLE_TRAIN_EPOCHS", "2")),
        per_device_train_batch_size=int(os.getenv("PURPLE_TRAIN_BATCH_SIZE", "1")),
        gradient_accumulation_steps=int(os.getenv("PURPLE_TRAIN_GRAD_ACCUM", "8")),
        learning_rate=float(os.getenv("PURPLE_TRAIN_LR", "0.0002")),
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        fp16=False,
        bf16=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"LoRA adapter saved to: {OUTPUT_DIR}")
    print("Set MODEL_ID to this adapter directory when running the API, then test it before public use.")


if __name__ == "__main__":
    main()
