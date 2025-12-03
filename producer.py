# producer.py

import json
import pika
from analysis_common import build_samples

QUEUE = "tasks"


def main():
    print("[producer] Building sample list…")
    samples = build_samples()

    sample_name = "Data"
    file_list = samples[sample_name]["list"]

    print(f"[producer] {sample_name}: {len(file_list)} files")

    conn = pika.BlockingConnection(
        pika.ConnectionParameters("rabbitmq")
    )
    ch = conn.channel()
    ch.queue_declare(queue=QUEUE, durable=True)

    for url in file_list:
        msg = {"file_url": url, "sample": sample_name}
        ch.basic_publish(
            exchange="",
            routing_key=QUEUE,
            body=json.dumps(msg),
            properties=pika.BasicProperties(delivery_mode=2),
        )
        print(f"[producer] Queued {url}")

    conn.close()
    print("[producer] Done.")


if __name__ == "__main__":
    main()
