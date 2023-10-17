import pyterrier  as pt
from tqdm import tqdm
import pandas as pd
import click
import json
import time
import os
from collections import defaultdict
from evaluate import evaluate_list

@click.command()
@click.argument("dataset_folder")
@click.argument("at", type=int)
def main(dataset_folder, at):
    if not pt.started():
        pt.init()
    
    indexref = pt.IndexRef.of(f"./{dataset_folder}/terrier_index/")
    index = pt.IndexFactory.of(indexref)
    
    threshold_for_at = {10:99999, 100:99999, 1000:99999, 10000:200, 100000:60}

    qrels = defaultdict(dict)
    run = {}
    with open(f"{dataset_folder}/relevant_pairs.jsonl") as f:
        for i, q_data in enumerate(map(json.loads, f)):
            run[q_data["id"]] = q_data["question"]
            qrels[q_data["id"]][q_data["doc_id"]] = 1
            
            if i>threshold_for_at[at]:
                break
            
    question_ids, questions = list(zip(*run.items()))

    # for large ats, pyterrier would be to slow
    # so we for now just run on a small question subset
    
    with open(f"results/pyterrier.csv", "a") as fOut:
        questions = questions#[:200]

        bm25_pipe = pt.rewrite.tokenise() >> pt.BatchRetrieve(index, wmodel="BM25", num_results=at)#).parallel(3)
        
        results = []
        time_list = []
        st = time.time()
        
        for question in tqdm(questions):
            
            questions_dataframe = pd.DataFrame([{"qid":0, "query":question.lower()}])

            df_results = bm25_pipe.transform(questions_dataframe)
            
            results.append((df_results["docno"], df_results["score"]))
        
        end_t = time.time()
        
        #correct format
        results = [list(zip(r[0].tolist(), r[1].tolist())) for r in results]
        
        r_evaluate = evaluate_list(qrels, results, question_ids)
        print(r_evaluate)
        fOut.write(f"{dataset_folder},{at},{len(questions)/(end_t-st)},{r_evaluate['ndcg@10']},{r_evaluate['ndcg@1000']},{r_evaluate['recall@1000']}\n")


if __name__=="__main__":
    main()  