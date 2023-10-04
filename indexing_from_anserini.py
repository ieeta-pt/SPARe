from collection import SparseCollection, SparseCollectionCSR
from backend import TYPE
import json
from text2vec import BagOfWords
import torch
from collections import defaultdict
from tqdm import tqdm
from weighting_model import BM25Transform
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
    
    index_reader = IndexReader(f"{msmarco_folder}/anserini_index")
    analyzer = Analyzer(get_lucene_analyzer())
    print("build token2id dict")
    token2id = {term.term:i for i,term in enumerate(index_reader.terms())}
    
    def tokenizer(text):
        tokens_ids = []
        for token in analyzer.analyze(text.lower()):
            #token_id=token2id[token]
            tokens_ids.append(token2id[token])
            #if token in token2id:
            #    token_id=token2id[token]
            #    if token_id is not None:
            #        tokens_ids.append(token_id)
        return tokens_ids

    bow = BagOfWords(tokenizer, len(token2id))

    PATH_TO_MSMARCO = list(glob.glob(f"{msmarco_folder}/corpus*"))[0]
    
    if max_lines==-1:
        max_lines = int(re.findall(r"_L([0-9]+)", PATH_TO_MSMARCO)[0])        
    
    def get_title_abstract(string):
        data = json.loads(string)
        title, abstract = data["title"], data["abstract"]
        return f"{title} {abstract}"

    with open(PATH_TO_MSMARCO) as f:
        collection_iterator = enumerate(map(get_title_abstract,f))
        
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