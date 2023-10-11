import pyterrier  as pt
from tqdm import tqdm
import pandas as pd
import click
import json
import time
import os

@click.command()
@click.argument("dataset_folder")
@click.argument("at", type=int)
def main(dataset_folder, at):
    if not pt.started():
        pt.init()
    
    indexref = pt.IndexRef.of(f"./{dataset_folder}/terrier_index/")
    index = pt.IndexFactory.of(indexref)
    

    with open(f"{dataset_folder}/relevant_pairs.jsonl") as f:
        questions = list({line["question"] for line in map(json.loads, f)})

    # for large ats, pyterrier would be to slow
    # so we for now just run on a small question subset
    
    with open(f"results/pyterrier.csv", "a") as fOut:
        questions = questions[:200]

        bm25_pipe = pt.rewrite.tokenise() >> pt.BatchRetrieve(index, wmodel="BM25", num_results=at)#).parallel(3)
        
        results = []
        time_list = []
        st = time.time()
        
        for question in tqdm(questions):
            
            questions_dataframe = pd.DataFrame([{"qid":0, "query":question.lower()}])

            df_results = bm25_pipe.transform(questions_dataframe)
            
            results.append(df_results["docno"].tolist())
        
        fOut.write(f"{dataset_folder},{at},{len(questions)/(time.time()-st)}\n")


if __name__=="__main__":
    main()  