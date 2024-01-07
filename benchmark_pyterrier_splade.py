import pyterrier  as pt
from tqdm import tqdm
import pandas as pd
import click
import json
import time
import os
from collections import defaultdict
from evaluate import evaluate_list

def _matchop(t, w):
    import base64
    import string
    if not all(a in string.ascii_letters + string.digits for a in t):
        encoded = base64.b64encode(t.encode('utf-8')).decode("utf-8") 
        t = f'#base64({encoded})'
    if w != 1:
        t = f'#combine:0={w}({t})'
    return t

@click.command()
@click.argument("dataset_folder")
@click.argument("at", type=int)
def main(dataset_folder, at):
    if not pt.started():
        pt.init()
    
    index_name = f"./{dataset_folder}/splade_pyterrier_index/"
    
    #threshold_for_at = {10:99999, 100:99999, 1000:99999, 10000:200, 100000:60}

    threshold_for_at = {10:99999, 100:99999, 1000:1000, 10000:200, 100000:60}
    
    qrels = defaultdict(dict)
    with open(f"{dataset_folder}/relevant_pairs.jsonl") as f:
        for i, q_data in enumerate(map(json.loads, f)):
            qrels[q_data["id"]][q_data["doc_id"]] = 1
            

    # for large ats, pyterrier would be to slow
    # so we for now just run on a small question subset
    
    questions_data = []
    with open(f"{dataset_folder}/splade_msmarco_questions_bow.jsonl") as f:
        for q in map(json.loads, f):
            
            query = ' '.join( _matchop(k, v * 100) for k, v in sorted(q["bow"].items(), key=lambda x: (-x[1], x[0])))
            
            questions_data.append({
                "qid":q["docno"], 
                "query": query,
            })

    questions_data = questions_data[:threshold_for_at[at]]
    #questions_ids = questions_ids[:threshold_for_at[at]]
    
    questions_dataframe = pd.DataFrame(questions_data)
    
    with open(f"results/pyterrier_splade.csv", "a") as fOut:

        retr_pipe = pt.BatchRetrieve(index_name, wmodel="Tf", num_results=at)#).parallel(3)
        
        results = []
        time_list = []
        st = time.time()
        
        #print(questions_dataframe)
        df_results = retr_pipe.transform(questions_dataframe)
        #print(df_results)
        questions_results = defaultdict(list)
        df_results.groupby("qid").apply(lambda x: questions_results[x.iloc[0]["qid"]].extend(zip(x["docno"], x["score"])))
        questions_ids, results = zip(*questions_results.items())
        
        end_t = time.time()
        
        #correct format
        #results = [list(zip(r[0].tolist(), r[1].tolist())) for r in results]
        
        r_evaluate = evaluate_list(qrels, results, questions_ids)
        print(r_evaluate)
        fOut.write(f"{dataset_folder},{at},{len(questions_data)/(end_t-st)},{r_evaluate['ndcg@10']},{r_evaluate['ndcg@1000']},{r_evaluate['recall@1000']}\n")


if __name__=="__main__":
    main()  