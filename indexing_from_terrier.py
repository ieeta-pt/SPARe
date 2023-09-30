from collection import SparseCollection, SparseCollectionCSR
from backend import TYPE
import json
from text2vec import BagOfWords
import torch
from collections import defaultdict
from tqdm import tqdm
from weighting_model import BM25Transform
import click
import pyterrier  as pt
import pandas as pd
import glob
import re

@click.command()
@click.argument("msmarco_folder")
@click.option("--max_lines", default=-1)
def main(msmarco_folder, max_lines):
    pt.init()

    indexref = pt.IndexRef.of(f"{msmarco_folder}/terrier_index")
    index = pt.IndexFactory.of(indexref)

    def tp_func():
        stops = pt.autoclass("org.terrier.terms.Stopwords")(None)
        stemmer = pt.autoclass("org.terrier.terms.PorterStemmer")(None)
        def _apply_func(row):
            words = row["query"].split(" ") # this is safe following pt.rewrite.tokenise()
            words = [stemmer.stem(w) for w in words if not stops.isStopword(w) ]
            return words#" ".join(words)
        return _apply_func 

    pipe = pt.rewrite.tokenise() >> pt.apply.query(tp_func())
    token2id = {word.getKey():i for i,word in enumerate(index.getLexicon()) }

    vocab_size = len(index.getLexicon())

    def tokenizer(text):
        tokens_ids = []
        for token in pipe(pd.DataFrame([{"qid":0, "query":text.lower()}]))["query"][0]:
            if token in token2id:
                token_id=token2id[token]
                if token_id is not None:
                    tokens_ids.append(token_id)
        return tokens_ids

    bow = BagOfWords(tokenizer, vocab_size)

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

    sparseCSR_collection.save_to_file(f"csr_msmarco")

    sparseCSR_collection.transform(BM25Transform(k1=1.2, b=0.75))

    sparseCSR_collection.save_to_file(f"csr_msmarco_bm25_12_075")


if __name__ == '__main__':
    main()