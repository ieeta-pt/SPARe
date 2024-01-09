from spare.collection import SparseCollection, SparseCollectionCSR
from spare.retriever import SparseRetriever
from spare.weighting_model import BM25WeightingModel
from spare import TYPE
import json

from spare.text2vec import BagOfWords

from spare.transformations import DtypeTransform

import pyterrier  as pt
import pandas as pd

import time 
import click
from collections import defaultdict

from evaluate import evaluate_spare

@click.command()
@click.argument("dataset_folder")
@click.argument("at", type=int)
@click.option("--cache_bow", is_flag=True)
@click.option("--fp_16", is_flag=True)
@click.option("--objective", default="accuracy")
@click.option("--algorithm", default="dot")
def main(dataset_folder, at, cache_bow, fp_16, objective, algorithm):
    notes = f"_{objective}_{algorithm}"
    print("Dataset", dataset_folder)
    if not pt.started():
        pt.init()

    indexref = pt.IndexRef.of(f"./{dataset_folder}/terrier_index/")
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

    


    #mp.set_start_method("spawn")
    
    # load collection data and metadata from a previously created folder
    
    print("load collection")    
    sparse_collection = SparseCollectionCSR.load_from_file(f"{dataset_folder}/csr_terrier_bm25_12_075") # cpu
    print("dataset size", sparse_collection.shape)
    # creates the bow function to convert text into vec
    
    if fp_16:
        sparse_collection.transform(DtypeTransform(TYPE.float16))
        notes += "_fp_16"
    
    print("dtype values", sparse_collection.sparse_vecs[-1].dtype)
    
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
    
    sparse_retriver = SparseRetriever(sparse_collection, bow, BM25WeightingModel(), objective=objective, algorithm=algorithm)

    # load a large number of questions
    print("Num questions", len(questions))
    
    #questions = questions[:10000]
        
    s = time.time()
    # Retrieve by default utilizes the maximum amount of resources available
    out = sparse_retriver.retrieve(questions, top_k=at, return_scores=True, profiling=True) # TODO load directly to the device
    times = out.timmings
    e = time.time()
    
    print("computing metrics")
    r_evaluate = evaluate_spare(qrels, out, question_ids)
    #print(r_evaluate)
    #"recall@1000", "ndcg@10", "ndcg@10000"
    
    with open(f"results/sparse_retriever_from_pyterrier_devices_{'-'.join(sparse_collection.backend.devices)}{notes}.csv", "a") as fOut:
        print("Total retrieve time", (e-s), "QPS", len(questions)/(e-s))
        fOut.write(f"{dataset_folder},{at},{len(questions)/(e-s)},{r_evaluate['ndcg@10']},{r_evaluate['ndcg@1000']},{r_evaluate['recall@1000']},{times[0]},{times[1]}\n")
    
if __name__=="__main__":
    main()