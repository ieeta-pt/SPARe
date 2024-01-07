from tqdm import tqdm
import click
import json
import time
import os
from collections import defaultdict
from pyserini.search.lucene import LuceneImpactSearcher
from pyserini.encode import QueryEncoder

class PreEncoder(QueryEncoder):
    
    def encode(self, text, **kwargs):
        # text will be in bow format as a dict
        if not isinstance(text, list):
            text = [text]
        return self._get_encoded_query_token_wight_dicts(text)[0]
    
    def _get_encoded_query_token_wight_dicts(self, tok_weights):
        to_return = []
        for _tok_weight in tok_weights:
            _weights = {}
            for token, weight in _tok_weight.items():
                weight_quanted = round(weight / 5 * 256)
                _weights[token] = weight_quanted
            to_return.append(_weights)
        return to_return
    
from evaluate import evaluate_list

@click.command()
@click.argument("at", type=int)
@click.option("--threads", type=int, default=1)
def main(at, threads):
    
    searcher = LuceneImpactSearcher(f"beir_datasets/msmarco/deepct_pyserini_index", 
                                query_encoder= PreEncoder())
    
    threshold_for_at = {10:99999, 100:99999, 1000:99999, 10000:200, 100000:60}   
    #threshold_for_at = {10:10, 100:10, 1000:10, 10000:10, 100000:10}  
     
    qrels = defaultdict(dict)
    with open(f"beir_datasets/msmarco/relevant_pairs.jsonl") as f:
        for i,q_data in enumerate(map(json.loads, f)):
            qrels[q_data["id"]][q_data["doc_id"]] = 1
            
            if i>threshold_for_at[at]:
                break
            
    questions = []
    questions_ids = []
    with open(f"beir_datasets/msmarco/deepct_questions_bow.jsonl") as f:
        for q in map(json.loads, f):
            questions.append(q["bow"])
            questions_ids.append(q["docno"])

    questions = questions[:threshold_for_at[at]]
    questions_ids = questions_ids[:threshold_for_at[at]]

    # for large ats, pyterrier would be to slow
    # so we for now just run on a small question subset
    
    #questions = questions#[:400]
    
    results = []
    time_list = []
    st = time.time()
    
    if threads==1:
        for question in tqdm(questions):

            hits = searcher.search(question, k=at)
            
            results.append(list(map(lambda x:(x.docid, x.score), hits)))
    else:
        print("run batch search")
        #questions = questions[:30]
        
        #q_ids = list(map(str, range(len(questions))))
        hits = searcher.batch_search(questions, questions_ids, k=at, threads=threads)
        #for x in hits:
        #    print(x.values)
        #    break
        #print(hits)
        for q_id in questions_ids:
            results.append(list(map(lambda x:(x.docid, x.score), hits[q_id])))
    end_t = time.time()
    
    r_evaluate = evaluate_list(qrels, results, questions_ids)
    
    out_file_name = "pyserini_deepct"
    if threads>1:
        out_file_name += f"_mp{threads}"
    
    with open(f"results/{out_file_name}.csv", "a") as fOut:
        fOut.write(f"{dataset_folder},{at},{len(questions)/(end_t-st)},{r_evaluate['ndcg@10']},{r_evaluate['ndcg@1000']},{r_evaluate['recall@1000']}\n")
    

if __name__=="__main__":
    main()  