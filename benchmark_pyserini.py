import pyterrier  as pt
from tqdm import tqdm
import pandas as pd
import click
import json
import time
import os
from pyserini.search.lucene import LuceneSearcher

@click.command()
@click.argument("dataset_folder")
@click.argument("at", type=int)
def main(dataset_folder, at):
    
    searcher = LuceneSearcher(f"{dataset_folder}/anserini_index")
    searcher.set_bm25(1.2, 0.75)

    with open(f"{dataset_folder}/relevant_pairs.jsonl") as f:
        questions = list({line["question"] for line in map(json.loads, f)})

    # for large ats, pyterrier would be to slow
    # so we for now just run on a small question subset
    with open(f"results/pyseniri.csv", "a") as fOut:
        questions = questions[:400]
        
        results = []
        time_list = []
        st = time.time()
        
        for question in tqdm(questions):
            

            hits = searcher.search(question.lower(), k=at)
            
            results.append(list(map(lambda x:x.docid, hits)))
        
        fOut.write(f"{dataset_folder},{at},{len(questions)/(time.time()-st)}\n")
    

if __name__=="__main__":
    main()  