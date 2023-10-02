from collection import SparseCollection, SparseCollectionCSR
from retriever import SparseRetriever
from weighting_model import BM25WeightingModel
from backend import TYPE
import json
import psutil
from text2vec import BagOfWords
import torch
from collections import defaultdict
from tqdm import tqdm
import multiprocessing  as mp

import pyterrier  as pt
import pandas as pd

import torch.nn as nn
import time 
pt.init()

indexref = pt.IndexRef.of("../syn-question-col-analysis/datasets/msmarco/terrier_index/")
index = pt.IndexFactory.of(indexref)

def tp_func():
    stops = pt.autoclass("org.terrier.terms.Stopwords")(None)
    stemmer = pt.autoclass("org.terrier.terms.PorterStemmer")(None)
    def _apply_func(row):
        words = row["query"].split(" ") # this is safe following pt.rewrite.tokenise()
        words = [stemmer.stem(w) for w in words if not stops.isStopword(w) ]
        return words
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



if __name__=="__main__":
    #mp.set_start_method("spawn")
    
    sparse_collection = SparseCollectionCSR.load_from_file("csr_msmarco_bm25_12_075_terrier")

    bow = BagOfWords(tokenizer, vocab_size)

    # add the option if no weighitngmodel then use the collection weighting model
    sparse_retriver = SparseRetriever(sparse_collection, lambda x:x, BM25WeightingModel())

    with open("../syn-question-col-analysis/question_generation/gen_output/msmarco/selected_corpus_lm_fcm_STD2_L10000_gpt-neo-1.3B_BS_5_E13931.459746599197.jsonl") as f:
        questions = [line["question"] for line in map(json.loads, f)]
    
    s_bow = time.time()
    questions = [bow(q) for q in questions]
    e_bow = time.time()
    
    s = time.time()
    sparse_retriver.retrieve(questions)
    e = time.time()
    
    print("Total time", (e-s), "QPS", len(questions)/(e-s), "Bow time:", (e_bow-s_bow))