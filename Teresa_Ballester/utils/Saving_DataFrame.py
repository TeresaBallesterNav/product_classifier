import pandas as pd
import gzip
import json
from tqdm import tqdm
from pathlib import Path

def parse(path):
    with gzip.open(path, 'r') as g:
        for line in g:
            yield json.loads(line)

def save_in_chunks(input_path, output_folder, chunk_size=100_000):
    data = []
    chunk_index = 0

    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for i, item in enumerate(tqdm(parse(input_path), desc="Reading Data")):
        data.append(item)

        if (i + 1) % chunk_size == 0:
            df = pd.DataFrame(data)
            df.to_parquet(output_folder / f"products_chunk_{chunk_index}.parquet", index=False)
            print(f"Chunk {chunk_index} saved")
            data = []
            chunk_index += 1

    if data:
        df = pd.DataFrame(data)
        df.to_parquet(output_folder / f"products_chunk_{chunk_index}.parquet", index=False)
        print(f"Final chunk {chunk_index} saved")

    print(f"Finished saving to {output_folder}")

# Ejecutar
if __name__ == "__main__":
    input_path = "/home/Teresa_Ballester/product-classifier/data/amz_products_small.jsonl.gz"
    output_dir = "/home/Teresa_Ballester/product-classifier/data/processed_data"
    save_in_chunks(input_path, output_dir)

