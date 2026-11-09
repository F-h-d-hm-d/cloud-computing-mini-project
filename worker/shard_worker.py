
import os
import json
import boto3
from collections import Counter

ENDPOINT = os.getenv("S3_ENDPOINT", "http://seaweedfs.data.svc.cluster.local:8333")
SHARD_COUNT = int(os.getenv("SHARD_COUNT", "2"))
SHARD_ID = int(os.getenv("SHARD_ID", "0"))

if SHARD_COUNT < 1 or not 0 <= SHARD_ID < SHARD_COUNT:
    raise ValueError("Invalid shard configuration")

s3 = boto3.client(
    "s3",
    endpoint_url=ENDPOINT,
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "admin"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "admin"),
    region_name="us-east-1",
)

source = s3.get_object(
    Bucket="datasets",
    Key="teacher_3000.jsonl",
)["Body"].read().decode("utf-8-sig")

records = []
seen = set()
duplicates = 0
invalid = 0

for index, line in enumerate(source.splitlines()):
    if index % SHARD_COUNT != SHARD_ID or not line.strip():
        continue

    record = json.loads(line)
    command_id = record["id"]

    if command_id in seen:
        duplicates += 1
        continue

    seen.add(command_id)
    records.append(record)

    if record.get("valid") is not True:
        invalid += 1

key = f"shards/shard_{SHARD_ID}.jsonl"
body = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)

s3.put_object(
    Bucket="datasets",
    Key=key,
    Body=body.encode("utf-8"),
)

print(json.dumps({
    "shard_id": SHARD_ID,
    "shard_count": SHARD_COUNT,
    "records": len(records),
    "invalid_labels": invalid,
    "duplicate_ids_within_shard": duplicates,
    "output": key,
}, indent=2))
