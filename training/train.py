
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

MODEL = "EleutherAI/pythia-70m"
DATA = Path("data/train_1500.jsonl")
OUTPUT = Path("models/v2")

torch.set_num_threads(6)

tokenizer = AutoTokenizer.from_pretrained(MODEL)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.config.pad_token_id = tokenizer.pad_token_id

class CommandDataset(Dataset):
    def __init__(self):
        self.examples = []
        for line in DATA.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            prompt = f'Command: {row["text"]}\nJSON: '
            answer = json.dumps(row["label"], ensure_ascii=False)
            prefix = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            suffix = tokenizer(
                answer + tokenizer.eos_token,
                add_special_tokens=False
            )["input_ids"]

            ids = (prefix + suffix)[:192]
            labels = ([-100] * len(prefix) + suffix)[:192]

            self.examples.append({
                "input_ids": ids,
                "attention_mask": [1] * len(ids),
                "labels": labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

def collate(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    return {
        "input_ids": torch.tensor([
            x["input_ids"] + [tokenizer.pad_token_id] *
            (max_len - len(x["input_ids"])) for x in batch
        ]),
        "attention_mask": torch.tensor([
            x["attention_mask"] + [0] *
            (max_len - len(x["attention_mask"])) for x in batch
        ]),
        "labels": torch.tensor([
            x["labels"] + [-100] *
            (max_len - len(x["labels"])) for x in batch
        ])
    }

def main():
    if not DATA.exists():
        raise FileNotFoundError(DATA)

    dataset = CommandDataset()
    print("Training examples:", len(dataset), flush=True)

    args = TrainingArguments(
        output_dir=str(OUTPUT),
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        use_cpu=True,
        dataloader_num_workers=0,
        seed=42
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collate
    )

    trainer.train()
    trainer.save_model(str(OUTPUT))
    tokenizer.save_pretrained(str(OUTPUT))

    history = [
        {"step": entry["step"], "loss": entry["loss"]}
        for entry in trainer.state.log_history
        if "loss" in entry
    ]

    with (OUTPUT / "loss.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss"])
        writer.writeheader()
        writer.writerows(history)

    print("Training completed!")
    print("Model saved to:", OUTPUT)
    print("Loss saved to:", OUTPUT / "loss.csv")

if __name__ == "__main__":
    main()
