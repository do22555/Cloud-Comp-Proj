# producer.py

import json
import pika

from analysis_common import build_samples, SAMPLE_DEFS

QUEUE = "tasks"

# nice easy one this: just grab the files and put them in a round robin queue
def main():
    print("[producer] Building sample list…")
    samples = build_samples()

    conn = pika.BlockingConnection(pika.ConnectionParameters("rabbitmq"))
    ch = conn.channel()
    ch.queue_declare(queue=QUEUE, durable=True)

    total_files = 0

    # Loop over all defined samples (Data, backgrounds, signal)
    for sample_name in SAMPLE_DEFS.keys():
        file_list = samples[sample_name]["list"]
        print(f"[producer] {sample_name}: {len(file_list)} files")
        total_files += len(file_list)

        for url in file_list:
            msg = {"file_url": url, "sample": sample_name}
            ch.basic_publish(
                exchange="",
                routing_key=QUEUE,
                body=json.dumps(msg),
                properties=pika.BasicProperties(delivery_mode=2),
            )
            print(f"[producer] Queued {url} ({sample_name})")

    conn.close()
    print(f"[producer] Done. Queued {total_files} files.")


if __name__ == "__main__":
    main()
