from spare.collection import SparseCollection, SparseCollectionCSR
from spare import TYPE
import json
from spare.text2vec import BagOfWords
import torch
from collections import defaultdict
from tqdm import tqdm
from spare.weighting_model import BM25Transform
import click
from pyserini.index.lucene import IndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer

import glob
import re
import os
@click.command()
@click.argument("msmarco_folder")
@click.option("--max_lines", default=-1)
def main(msmarco_folder, max_lines):
    
    if os.path.exists(os.path.join(msmarco_folder,f"csr_anserini_bm25_12_075")):
        print(f"Skip {msmarco_folder}, already exists")
        return 0
    
    index_reader = IndexReader(f"{msmarco_folder}/anserini_index")

    print("build token2id dict")
    token2id = {term.term:i for i,term in enumerate(index_reader.terms())}
    
    def tokenizer(text):
        tokens_ids = []
        for token in index_reader.analyze(text.lower()):
            #token_id=token2id[token]
            if token in token2id:
                tokens_ids.append(token2id[token])
            #if token in token2id:
            #    token_id=token2id[token]
            #    if token_id is not None:
            #        tokens_ids.append(token_id)
        return tokens_ids

    bow = BagOfWords(tokenizer, len(token2id))

    PATH_TO_MSMARCO = list(glob.glob(f"{msmarco_folder}/collection/corpus*"))[0]
    
    if max_lines==-1:
        max_lines = int(re.findall(r"_L([0-9]+)", PATH_TO_MSMARCO)[0])        
    
    def get_id_contents(string):
        data = json.loads(string)
        #text = data["id"], data["contents"]
        return data["id"], data["contents"]

    with open(PATH_TO_MSMARCO) as f:
        collection_iterator = map(get_id_contents,f)
        
        sparseCSR_collection = SparseCollectionCSR.from_text_iterator(collection_iterator,
                                                                        collection_maxsize=max_lines,
                                                                        text_to_vec=bow,
                                                                        dtype=TYPE.float32,
                                                                        indices_dtype=TYPE.int32,
                                                                        backend="torch") 

    #sparseCSR_collection.save_to_file(f"csr_anserini")

    sparseCSR_collection.transform(BM25Transform(k1=1.2, b=0.75))

    sparseCSR_collection.save_to_file(os.path.join(msmarco_folder,f"csr_anserini_bm25_12_075"))


if __name__ == '__main__':
    main()