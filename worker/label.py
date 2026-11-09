
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

INPUT = Path("data/commands_pool.jsonl")
OUTPUT = Path("results/teacher_3000.jsonl")
SCHEMA = Path("data/schema.json")

MODEL = "phi3.5"
URL = "http://localhost:11434/api/generate"
MAX_RETRIES = 3

FIELDS = ("intent", "date", "time", "place")


def load_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_allowed_intents(schema):
    # Read common schema layouts without inventing intent names.
    candidates = []

    def visit(obj, parent=""):
        if isinstance(obj, dict):
            for key, value in obj.items():
                name = key.lower()

                if name in ("intents", "allowed_intents", "intent_labels"):
                    if isinstance(value, list):
                        candidates.extend(
                            x for x in value if isinstance(x, str)
                        )

                if name == "intent" and isinstance(value, dict):
                    enum = value.get("enum")
                    if isinstance(enum, list):
                        candidates.extend(
                            x for x in enum if isinstance(x, str)
                        )

                visit(value, name)

        elif isinstance(obj, list):
            for value in obj:
                visit(value, parent)

    visit(schema)
    return sorted(set(candidates))


def make_prompt(command, allowed_intents):
    intent_instruction = (
        "Allowed intent values: "
        + ", ".join(allowed_intents)
        if allowed_intents
        else "Use the intent names defined by the provided schema."
    )

    return f"""You are a command-parsing assistant.

Extract information from the command.

Return exactly one JSON object with these four keys:
intent, date, time, place

Rules:
- Return JSON only.
- Use null when information is missing.
- Never invent missing dates, times, or places.
- Preserve date, time, and place expressions from the command.
- Do not include explanations or markdown.
- {intent_instruction}

Command: {command}

JSON:
"""


def request_label(command, allowed_intents):
    payload = {
        "model": MODEL,
        "prompt": make_prompt(command, allowed_intents),
        "stream": False,
        "format": "json",
        "keep_alive": "15m",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_ctx": 1024,
            "num_gpu": 999,
            "num_batch": 32,
            "num_predict": 128
        }
    }

    request = urllib.request.Request(
        URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)

    return result["response"]


def validate(raw, allowed_intents):
    try:
        result = json.loads(raw)
    except (ValueError, TypeError):
        return None

    if not isinstance(result, dict):
        return None

    if set(result) != set(FIELDS):
        return None

    if not isinstance(result["intent"], str):
        return None

    if allowed_intents and result["intent"] not in allowed_intents:
        return None

    for field in ("date", "time", "place"):
        if result[field] is not None and not isinstance(result[field], str):
            return None

    return result


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    if not SCHEMA.exists():
        raise FileNotFoundError(SCHEMA)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    commands = load_jsonl(INPUT)
    schema = json.loads(SCHEMA.read_text(encoding="utf-8-sig"))
    allowed_intents = extract_allowed_intents(schema)

    print("Total commands:", len(commands))
    print("Allowed intents:", allowed_intents or "NOT DETECTED")

    if not allowed_intents:
        raise ValueError(
            "Could not detect allowed intents in schema.json. "
            "Stop and check the schema before labelling."
        )

    completed = set()

    if OUTPUT.exists():
        for row in load_jsonl(OUTPUT):
            if row.get("valid") is True:
                completed.add(row["id"])

    print("Already completed:", len(completed), flush=True)

    with OUTPUT.open("a", encoding="utf-8") as output:
        for index, command in enumerate(commands, 1):
            command_id = command["id"]

            if command_id in completed:
                continue

            raw = ""
            parsed = None
            error = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    raw = request_label(
                        command["text"], allowed_intents
                    )

                    parsed = validate(raw, allowed_intents)

                    if parsed is not None:
                        break

                    error = "Invalid JSON or schema values"

                except (
                    urllib.error.URLError,
                    TimeoutError,
                    KeyError,
                    ValueError
                ) as exc:
                    error = str(exc)

                print(
                    f"Retry {attempt}/{MAX_RETRIES} "
                    f"for {command_id}: {error}",
                    flush=True
                )
                time.sleep(2)

            record = {
                "id": command_id,
                "text": command["text"],
                "teacher_raw": raw,
                "valid": parsed is not None
            }

            if parsed is not None:
                record["label"] = parsed
                completed.add(command_id)
            else:
                record["error"] = error

            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )
            output.flush()

            print(
                f"Processed {index}/{len(commands)} "
                f"| Valid: {len(completed)} "
                f"| Current: {'OK' if parsed else 'FAILED'}",
                flush=True
            )

    print("\nLabelling pass completed!")
    print("Valid labels:", len(completed))
    print("Output:", OUTPUT)
    print(
        "Note: Valid means structurally valid; "
        "semantic accuracy still needs checking."
    )


if __name__ == "__main__":
    main()
