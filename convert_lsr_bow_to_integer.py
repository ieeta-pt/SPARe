
import click
import glob
from tqdm import tqdm
import json
import re
import os

@click.command()
@click.argument("dataset_folder")
@click.argument("lsr_name")
def main(dataset_folder, lsr_name):
    
    path_to_dataset = f"{dataset_folder}/collection_{lsr_name}_bow/collection_bow.jsonl"
    
    corpus_path = list(glob.glob(f"{dataset_folder}/collection/corpus*"))[0]
    max_lines = int(re.findall(r"_L([0-9]+)", corpus_path)[0])    
    
    def load_collection():

        with open(path_to_dataset) as f:
            for article in tqdm(map(json.loads, f), total=max_lines):
                yield article

    if not os.path.exists(f"{dataset_folder}/collection_{lsr_name}_integer"):
        os.makedirs(f"{dataset_folder}/collection_{lsr_name}_integer")
    
    with open(f"{dataset_folder}/collection_{lsr_name}_integer/collection_vector_integer.jsonl", "w") as f:
        for article in load_collection():
            data = {
                "id": article["docno"],
                "vector": {k:int(v*100) for k,v in article["bow"].items()},
            }
            f.write(f"{json.dumps(data)}\n")

if __name__=="__main__":
    main()  