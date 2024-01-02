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
    
    index = PisaIndex('./msmarco-passage-splade-pisa', stemmer='none')
    
    threshold_for_at = {10:99999, 100:99999, 1000:99999, 10000:99999, 100000:60}

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
    
    with open(f"results/pyterrier_pisa_splade.csv", "a") as fOut:
        #questions = questions[10]#[:200]
        index_searcher = index.quantized()
        #bm25 = index.bm25(k1=1.2, b=0.75, num_results=at, threads=threads)
        #bm25_pipe = pt.rewrite.tokenise() >> pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.k3" : 0}, num_results=at)#).parallel(3)
        
        results = []
        time_list = []
        print("read")
        questions_dataframe = pd.read_csv("msmarco-passage-splade-pisa/questions_tokinezed")[:100]
        #questions_dataframe = pd.DataFrame([{"qid":i, "query":question.lower()} for i,question in enumerate(questions)])
        
        print("start search")
        st = time.time()
        df_results = index_searcher.transform(questions_dataframe)
        end_s_t = time.time()
        print("end search")
        for i in range(len(questions_dataframe)):
            q_df_results = df_results[df_results["qid"]==i]
            results.append((q_df_results["docno"], q_df_results["score"]))
        end_t = time.time()
        
        print("SEARCH", end_s_t-st, "SEARCH QPS", len(questions_dataframe)/(end_s_t-st), "FIXING RESULTS", end_t-end_s_t)
        print("QPS", len(questions_dataframe)/(end_t-st))
        
        #correct format
        results = [list(zip(r[0].tolist(), r[1].tolist())) for r in results]
        
        r_evaluate = evaluate_list(qrels, results, question_ids)
        print(r_evaluate)
        fOut.write(f"{dataset_folder},{at},{len(questions_dataframe)/(end_t-st)},{r_evaluate['ndcg@10']},{r_evaluate['ndcg@1000']},{r_evaluate['recall@1000']}\n")


if __name__=="__main__":
    main()  