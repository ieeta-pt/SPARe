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
@click.option("--threads", type=int, default=1)
def main(dataset_folder, at, threads):
    
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
        
        if threads==1:
            for question in tqdm(questions):
    
                hits = searcher.search(question.lower(), k=at)
                
                results.append(list(map(lambda x:x.docid, hits)))
        else:
            print("run batch search")
            #questions = questions[:30]
            questions_text = list(map(lambda x:x.lower(), questions))
            q_ids = list(map(str, range(len(questions))))
            hits = searcher.batch_search(questions_text, q_ids, k=at, threads=threads)
            #for x in hits:
            #    print(x.values)
            #    break
            #print(hits)
            for i in q_ids:
                results.append(list(map(lambda x:x.docid, hits[i])))
        
        fOut.write(f"{dataset_folder},{at},{len(questions)/(time.time()-st)}\n")
    

if __name__=="__main__":
    main()  