from spare.collection import  SparseCollectionCSR
from spare.retriever import SparseRetriever
import json
import torch
from transformers import AutoTokenizer
import click
from collections import defaultdict
import time

from evaluate import evaluate_spare



@click.command()
@click.argument("dataset_folder")
@click.argument("at", type=int)
@click.option("--objective", default="accuracy")
@click.option("--algorithm", default="dot")
def main(dataset_folder, at, objective, algorithm):
    print("start")
    notes = f"_{objective}_{algorithm}"
    
    print("build token2id dict")
    tokenizer = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
    token2id = tokenizer.vocab
    
    print("load collection")
    sparse_collection = SparseCollectionCSR.load_from_file(f"{dataset_folder}/splade_spare_index")
    print("dataset size", sparse_collection.shape)

    print("load msmarco question converted with splade")
    questions = []
    questions_ids = []
    with open(f"{dataset_folder}/splade_questions_bow.jsonl") as f:
        for q in map(json.loads, f):
            questions.append({token2id[t]:v for t,v in q["bow"].items()})
            questions_ids.append(q["docno"])

    
    print("dtype values", sparse_collection.sparse_vecs[-1].dtype)
    # creates the bow function to convert text into vec
    
    # load a large number of questions
    qrels = defaultdict(dict)

    with open(f"{dataset_folder}/relevant_pairs.jsonl") as f:
        for q_data in map(json.loads, f):
            qrels[q_data["id"]][q_data["doc_id"]] = 1
    
    sparse_retriver = SparseRetriever(sparse_collection, objective=objective, algorithm=algorithm)

    print("Num questions", len(questions))
    
    #questions = questions[:10000]
        
    s = time.time()
    # Retrieve by default utilizes the maximum amount of resources available
    out = sparse_retriver.retrieve(questions, top_k=at, return_scores=True, profiling=True) # TODO load directly to the device
    times = out.timmings
    e = time.time()
    print("timings", times)
    print("computing metrics")
    r_evaluate = evaluate_spare(qrels, out, questions_ids)
    print(r_evaluate)
    
    with open(f"results/sparse_retriever_from_splade_devices_{'-'.join(sparse_collection.backend.devices)}{notes}.csv", "a") as fOut:
        print("Total retrieve time", (e-s), "QPS", len(questions)/(e-s))
        fOut.write(f"{dataset_folder},{at},{len(questions)/(e-s)},{r_evaluate['ndcg@10']},{r_evaluate['ndcg@1000']},{r_evaluate['recall@1000']},{times[0]},{times[1]}\n")
    
if __name__=="__main__":
    main()