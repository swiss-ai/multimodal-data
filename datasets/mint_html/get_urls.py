import multiprocessing
import os
import time

from datasets import load_dataset

NUM_PROCS = 32
BATCH_SIZE = 1000
DEST_DIR = "/tmp/vision/mint/parts"


def process_shard(rank, num_shards):
    ds = load_dataset(
        "mlfoundations/MINT-1T-HTML",
        revision="906a8b85cea61198ff7339c4dd711ad0b5361847",
        split="train",
        streaming=True,
        token=os.environ["HF_TOKEN"],
    ).select_columns(["images"])

    sharded_ds = ds.shard(num_shards=num_shards, index=rank)
    output_filename = os.path.join(DEST_DIR, f"urls_part_{rank:02d}.txt")

    print(f"Worker {rank} starting...")

    with open(output_filename, "w", encoding="utf-8") as f:
        for bi, batch in enumerate(sharded_ds.iter(batch_size=BATCH_SIZE)):
            valid_urls = [url for url_list in batch["images"] for url in url_list if url is not None]
            if valid_urls:
                f.write("\n".join(valid_urls) + "\n")

            if (bi + 1) % 1000 == 0:
                print(f"Worker {rank} processed {bi + 1} batches...")

    print(f"Worker {rank} finished!")


if __name__ == "__main__":
    print(f"Starting extraction with {NUM_PROCS} processes...")

    processes = []

    for rank in range(NUM_PROCS):
        p = multiprocessing.Process(target=process_shard, args=(rank, NUM_PROCS))
        p.start()
        processes.append(p)
        time.sleep(180)

    for p in processes:
        p.join()

    print("All workers done! You can now merge the files.")
