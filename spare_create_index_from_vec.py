from spare.collection import SparseCollectionCSR
import spare
import json

# to get splade tokenizer
import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
import pandas as pd
import pyt_splade
splade = pyt_splade.SpladeFactory()

token_to_id = { v:k for k,v in  splade.reverse_voc.items() }

VOCAB_SIZE = 30522


def load_bow():
    with open("splade_msmarco_bow.jsonl") as f:
        for article in map(json.loads, f):
            yield article["docno"], {token_to_id[token]:w for token, w in article["bow"].items()}

splade_collection = SparseCollectionCSR.from_vec_iterator(load_bow(),
                                                          8841823,
                                                          VOCAB_SIZE,
                                                          dtype=spare.float16)

splade_collection.save_to_file("splade_msmarco_index")