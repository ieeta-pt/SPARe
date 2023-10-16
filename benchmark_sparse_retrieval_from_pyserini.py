from spare.collection import SparseCollection, SparseCollectionCSR
from spare.retriever import SparseRetriever
from spare.weighting_model import BM25WeightingModel
from spare.backend import TYPE
import json
import psutil
from spare.text2vec import BagOfWords
import torch
from collections import defaultdict
from tqdm import tqdm
import multiprocessing  as mp
from spare.transformations import DtypeTransform

from evaluate import evaluate_spare

import pandas as pd

import time 
import click

from pyserini.index.lucene import IndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer

@click.command()
@click.argument("dataset_folder")
@click.argument("at", type=int)
@click.option("--cache_bow", is_flag=True)
@click.option("--fp_16", is_flag=True)
def main(dataset_folder, at, cache_bow, fp_16):
    print("Dataset", dataset_folder)
    notes = ""
    index_reader = IndexReader(f"{dataset_folder}/anserini_index")
    analyzer = Analyzer(get_lucene_analyzer())
    
    print("build token2id dict")
    token2id = {term.term:i for i,term in enumerate(index_reader.terms())}

    def tokenizer(text):
        tokens_ids = []
        for token in analyzer.analyze(text.lower()):
            #token_id=token2id[token]
            if token in token2id:
                tokens_ids.append(token2id[token])
            #if token in token2id:
            #    token_id=token2id[token]
            #    if token_id is not None:
            #        tokens_ids.append(token_id)
        return tokens_ids

    vocab_size = len(token2id)

    #mp.set_start_method("spawn")
    
    # load collection data and metadata from a previously created folder
    print("load collection")
    sparse_collection = SparseCollectionCSR.load_from_file(f"{dataset_folder}/csr_anserini_bm25_12_075") # cpu
    print("dataset size", sparse_collection.shape)
    # convert dtype
    if fp_16:
        sparse_collection.transform(DtypeTransform(TYPE.float16))
        notes += "_fp_16"
    print("dtype values", sparse_collection.sparse_vecs[-1].dtype)
    # creates the bow function to convert text into vec
    
    # load a large number of questions
    qrels = defaultdict(dict)
    run = {}
    with open(f"{dataset_folder}/relevant_pairs.jsonl") as f:
        for q_data in map(json.loads, f):
            run[q_data["id"]] = q_data["question"]
            qrels[q_data["id"]][q_data["doc_id"]] = 1
    
    question_ids, questions = list(zip(*run.items()))
    
    bow = BagOfWords(tokenizer, vocab_size)
    
    if cache_bow:
        questions = list(map(bow, questions))
        bow = lambda x:x
        notes += "_cache_bow"
    
    sparse_retriver = SparseRetriever(sparse_collection, bow, BM25WeightingModel())

    print("Num questions", len(questions))
    
    #questions = questions[:10000]
        
    s = time.time()
    # Retrieve by default utilizes the maximum amount of resources available
    out = sparse_retriver.retrieve(questions, top_k=at, return_scores=True, profiling=True) # TODO load directly to the device
    times = out.timmings
    e = time.time()
    
    r_evaluate = evaluate_spare(qrels, out, question_ids)
    
    with open(f"results/sparse_retriever_from_pyserini_devices_{'-'.join(sparse_collection.backend.devices)}{notes}.csv", "a") as fOut:
        print("Total retrieve time", (e-s), "QPS", len(questions)/(e-s))
        fOut.write(f"{dataset_folder},{at},{len(questions)/(e-s)},{r_evaluate['ndcg@10']},{r_evaluate['ndcg@10000']},{r_evaluate['recall@1000']},{times[0]},{times[1]}\n")
    
if __name__=="__main__":
    main()