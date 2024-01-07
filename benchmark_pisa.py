import pyterrier  as pt
from tqdm import tqdm
import pandas as pd
import click
import json
import time
import os
from collections import defaultdict
from evaluate import evaluate_list
from pyterrier_pisa import PisaIndex
@click.command()
@click.argument("dataset_folder")
@click.argument("at", type=int)
@click.option("--threads", type=int, default=1)
def main(dataset_folder, at, threads):
    if not pt.started():
        pt.init()
    
    index = PisaIndex(f"./{dataset_folder}/pisa_index/")
    
    threshold_for_at = {10:99999, 100:99999, 1000:99999, 10000:9999, 100000:60}

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
    
    out_file_name = "pyterrier_pisa"
    if threads>1:
        out_file_name += f"_mp{threads}"
    
    with open(f"results/{out_file_name}.csv", "a") as fOut:
        questions = questions#[:200]

        bm25 = index.bm25(k1=1.2, b=0.75, num_results=at, threads=threads)
        #bm25_pipe = pt.rewrite.tokenise() >> pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.k3" : 0}, num_results=at)#).parallel(3)
        
        results = []
        time_list = []
        
        
        questions_dataframe = pd.DataFrame([{"qid":question_ids[i], "query":question.lower()} for i,question in enumerate(questions)])
        
        st = time.time()
        df_results = bm25.transform(questions_dataframe)
        end_s_t = time.time()
        questions_results = defaultdict(list)
        #for i in range(len(questions)):
            #q_df_results = df_results[df_results["qid"]==i]
        #    q_df_results = df_results[i:i+at]
        #    results.append((q_df_results["docno"].tolist(), q_df_results["score"].tolist()))
        df_results.groupby("qid").apply(lambda x: questions_results[x.iloc[0]["qid"]].extend(zip(x["docno"], x["score"])))
        questions_ids, results = zip(*questions_results.items())
        
        end_t = time.time()
        
        print("SEARCH", end_s_t-st, "SEARCH QPS", len(questions)/(end_s_t-st), "FIXING RESULTS", end_t-end_s_t)
        print("QPS", len(questions)/(end_t-st))
        
        #correct format
        #results = [list(zip(*r)) for r in results]
        
        r_evaluate = evaluate_list(qrels, results, questions_ids)
        print(r_evaluate)
        fOut.write(f"{dataset_folder},{at},{len(questions)/(end_t-st)},{r_evaluate['ndcg@10']},{r_evaluate['ndcg@1000']},{r_evaluate['recall@1000']}\n")


if __name__=="__main__":
    main()  