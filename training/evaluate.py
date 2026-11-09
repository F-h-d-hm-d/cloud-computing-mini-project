
import csv
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE_MODEL = "EleutherAI/pythia-70m"
TRAINED_MODEL = "models/v2"
VALIDATION = Path("data/validation.jsonl")
RESULTS = Path("results")
FIELDS = ("intent", "date", "time", "place")

torch.set_num_threads(6)
RESULTS.mkdir(exist_ok=True)

def evaluate(model_path, name, examples):
    print(f"\nEvaluating {name}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    model.to("cpu")

    valid = 0
    exact = 0
    field_correct = 0
    field_total = len(examples) * len(FIELDS)
    predictions = []

    for index, example in enumerate(examples, 1):
        prompt = f'Command: {example["text"]}\nJSON: '
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        raw = tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

        parsed = None
        try:
            candidate = json.loads(raw)
            if isinstance(candidate, dict) and set(candidate) == set(FIELDS):
                parsed = candidate
                valid += 1
        except (ValueError, TypeError):
            pass

        expected = {key: example[key] for key in FIELDS}

        if parsed is not None:
            correct = sum(parsed[key] == expected[key] for key in FIELDS)
            field_correct += correct
            if correct == len(FIELDS):
                exact += 1

        predictions.append({
            "id": example["id"],
            "text": example["text"],
            "expected": expected,
            "prediction": parsed,
            "raw": raw
        })

        if index % 10 == 0:
            print(f"{name}: {index}/{len(examples)}", flush=True)

    summary = {
        "model": name,
        "total": len(examples),
        "json_validity_percent": round(100 * valid / len(examples), 2),
        "exact_match_percent": round(100 * exact / len(examples), 2),
        "field_accuracy_percent": round(
            100 * field_correct / field_total, 2
        )
    }

    (RESULTS / f"{name}_predictions.json").write_text(
        json.dumps(predictions, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    del model
    return summary

def main():
    examples = [
        json.loads(line)
        for line in VALIDATION.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    base = evaluate(BASE_MODEL, "untrained", examples)
    trained = evaluate(TRAINED_MODEL, "trained_v1", examples)

    summaries = [base, trained]

    with (RESULTS / "evaluation_v1.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    with (RESULTS / "evaluation_v1.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(base.keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\nEVALUATION RESULTS")
    print("-" * 55)
    for item in summaries:
        print(item["model"])
        print("  JSON validity:", item["json_validity_percent"], "%")
        print("  Exact match:", item["exact_match_percent"], "%")
        print("  Field accuracy:", item["field_accuracy_percent"], "%")

    print("\nSaved to results/evaluation_v1.json")

if __name__ == "__main__":
    main()
