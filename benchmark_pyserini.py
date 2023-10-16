from tqdm import tqdm
import click
import json
import time
import os
from collections import defaultdict
from pyserini.search.lucene import LuceneSearcher
from evaluate import evaluate_list

@click.command()
@click.argument("dataset_folder")
@click.argument("at", type=int)
@click.option("--threads", type=int, default=1)
def main(dataset_folder, at, threads):
    
    searcher = LuceneSearcher(f"{dataset_folder}/anserini_index")
    searcher.set_bm25(1.2, 0.75)

    qrels = defaultdict(dict)
    run = {}
    with open(f"{dataset_folder}/relevant_pairs.jsonl") as f:
        for q_data in map(json.loads, f):
            run[q_data["id"]] = q_data["question"]
            qrels[q_data["id"]][q_data["doc_id"]] = 1
            
    question_ids, questions = list(zip(*run.items()))


    # for large ats, pyterrier would be to slow
    # so we for now just run on a small question subset
    
    questions = questions#[:400]
    
    results = []
    time_list = []
    st = time.time()
    
    if threads==1:
        for question in tqdm(questions):

            hits = searcher.search(question.lower(), k=at)
            
            results.append(list(map(lambda x:(x.docid, x.score), hits)))
    else:
        print("run batch search")
        #questions = questions[:30]
        questions_text = list(map(lambda x:x.lower(), questions))
        #q_ids = list(map(str, range(len(questions))))
        hits = searcher.batch_search(questions_text, question_ids, k=at, threads=threads)
        #for x in hits:
        #    print(x.values)
        #    break
        #print(hits)
        for i in enumerate(question_ids):
            results.append(list(map(lambda x:(x.docid, x.score), hits[i])))
    end_t = time.time()
    
    r_evaluate = evaluate_list(qrels, results, question_ids)
    
    with open(f"results/pyseniri.csv", "a") as fOut:
        fOut.write(f"{dataset_folder},{at},{len(questions)/(end_t-st)}\n")
    

if __name__=="__main__":
    main()  