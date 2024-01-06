from spare.collection import SparseCollectionCSR
from spare.metadata import MetaDataDocID
import spare
import json

# to get splade tokenizer
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
import pandas as pd
from transformers import AutoTokenizer
import click
import glob
import re
from tqdm import tqdm


@click.command()
@click.argument("dataset_folder")
@click.argument("lsr_name")
def main(dataset_folder, lsr_name):

    path_to_dataset = f"{dataset_folder}/collection_{lsr_name}_bow/collection_bow.jsonl"
    
    corpus_path = list(glob.glob(f"{dataset_folder}/collection/corpus*"))[0]
    max_lines = int(re.findall(r"_L([0-9]+)", corpus_path)[0])      
    
    # BUILD VOCAB IS BETTER
    
    def load_bow():
        with open(path_to_dataset) as f:
            for article in map(json.loads, f):
                yield article["docno"], article["bow"]
    
    print("Build VOCAB")
    tokenizer_vocab = {}
    for _, bow in tqdm(load_bow()):
        for token in bow.keys():
            if token not in tokenizer_vocab:
                tokenizer_vocab[token] = len(tokenizer_vocab)
    
    def load_converted_bow():
        for docno, bow in load_bow():
            yield docno, {tokenizer_vocab[token]:w for token, w in bow.items()}
    #

    splade_collection = SparseCollectionCSR.from_vec_iterator(load_converted_bow(),
                                                            max_lines,
                                                            len(tokenizer_vocab),
                                                            dtype=spare.float16,
                                                            metadata=MetaDataDocID)

    splade_collection.save_to_file(f"{dataset_folder}/spare_{lsr_name}_index")

if __name__=="__main__":
    main()  