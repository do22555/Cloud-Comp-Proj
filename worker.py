# worker.py

import json
import hashlib
import re
from pathlib import Path

import numpy as np
import pika

from analysis_common import process_file

OUTPUT_DIR = Path("/data/output")


def slugify(name: str) -> str:
    return re.sub(r"\W+", "_", name)


def save_partial_result(result: dict):

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sample = result["sample"]
    file_url = result["file"]

    slug = slugify(sample)
    h = hashlib.sha1(file_url.encode()).hexdigest()[:10]

    out_path = OUTPUT_DIR / f"{slug}__{h}.npz"

    # --- SAVE A PURE NUMPY HISTOGRAM, NO Hist OBJECTS ---
    np.savez(
        out_path,
        edges=result["hist"]["edges"],
        values=result["hist"]["values"],
        variances=result["hist"]["variances"],
        events=result["events"],
        sample=sample,
        file_url=file_url,
    )

    print(f"[worker] Saved: {out_path}")


def on_message(ch, method, properties, body):
    message = json.loads(body)

    file_url = message["file_url"]
    sample = message["sample"]

    print(f"[worker] Processing {file_url}")

    try:
        result = process_file(file_url, sample)
        save_partial_result(result)
    except Exception as e:
        print(f"[worker] ERROR: {e}")

    ch.basic_ack(delivery_tag=method.delivery_tag)


def main():
    connection = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
    channel = connection.channel()

    channel.queue_declare(queue="tasks", durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue="tasks", on_message_callback=on_message)

    print("[worker] Waiting for tasks…")
    channel.start_consuming()


if __name__ == "__main__":
    main()
