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
    
    index = PisaIndex(f'{dataset_folder}/splade_pisa_index', stemmer='none')
    
    #threshold_for_at = {10:99999, 100:99999, 1000:99999, 10000:50, 100000:60}

    threshold_for_at = {10:1000, 100:50, 1000:99999, 10000:10, 100000:10}
    
    qrels = defaultdict(dict)
    with open(f"{dataset_folder}/relevant_pairs.jsonl") as f:
        for i, q_data in enumerate(map(json.loads, f)):
            qrels[q_data["id"]][q_data["doc_id"]] = 1
            

    # for large ats, pyterrier would be to slow
    # so we for now just run on a small question subset
    
    out_file_name = "pyterrier_pisa_splade"
    if threads>1:
        out_file_name += f"_mp{threads}"
    with open(f"results/{out_file_name}.csv", "a") as fOut:
        #questions = questions[10]#[:200]
        index_searcher = index.quantized(num_results=at, threads=threads)
        #bm25 = index.bm25(k1=1.2, b=0.75, num_results=at, threads=threads)
        #bm25_pipe = pt.rewrite.tokenise() >> pt.BatchRetrieve(index, wmodel="BM25", controls={"bm25.k3" : 0}, num_results=at)#).parallel(3)
        
        results = []
        time_list = []
        print("read")
        #questions_dataframe = pd.read_csv("msmarco-passage-splade-pisa/questions_tokinezed.csv")[:threshold_for_at[at]]
        questions_data = []
        with open(f"{dataset_folder}/splade_questions_bow.jsonl") as f:
            for q in map(json.loads, f):
                questions_data.append({
                    "qid":q["docno"], 
                    "query_toks": q["bow"],
                })

        questions_data = questions_data[:threshold_for_at[at]]
        #questions_ids = questions_ids[:threshold_for_at[at]]
        
        questions_dataframe = pd.DataFrame(questions_data)
        
        
        
        print("start search", len(questions_dataframe))
        st = time.time()
        df_results = index_searcher.transform(questions_dataframe)
        end_s_t = time.time()
        print("end search")
        
        questions_results = defaultdict(list)
        
        df_results.groupby("qid").apply(lambda x: questions_results[x.iloc[0]["qid"]].extend(zip(x["docno"], x["score"])))
        questions_ids, results = zip(*questions_results.items())
            #questions_ids.append(q_data["qid"])
            #q_df_results = df_results[df_results["qid"]==q_data["qid"]]
            #q_df_results = df_results[i:i+at+1]
            #for _,q_r in q_df_results.iterrows():
            #    assert q_r["qid"] == q_data["qid"]
            #results.append((q_df_results["docno"].tolist(), q_df_results["score"].tolist()))
        end_t = time.time()
        
        print("SEARCH", end_s_t-st, "SEARCH QPS", len(questions_dataframe)/(end_s_t-st), "FIXING RESULTS", end_t-end_s_t)
        print("QPS", len(questions_dataframe)/(end_t-st))
        
        #correct format
        #results = [list(zip(*r)) for r in results]
                
        r_evaluate = evaluate_list(qrels, results, questions_ids)
        print(r_evaluate)
        fOut.write(f"{dataset_folder},{at},{len(questions_dataframe)/(end_t-st)},{r_evaluate['ndcg@10']},{r_evaluate['ndcg@1000']},{r_evaluate['recall@1000']}\n")


if __name__=="__main__":
    main()  