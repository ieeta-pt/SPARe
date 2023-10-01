import pyterrier  as pt
from tqdm import tqdm
import pandas as pd
import time

if not pt.started():
    pt.init()
pt.set_property("index.meta.data-source", "fileinmem")
indexref = pt.IndexRef.of("../syn-question-col-analysis/datasets/msmarco/terrier_index/")
pt.set_property("index.meta.data-source", "fileinmem")
index = pt.IndexFactory.of(indexref)
pt.set_property("index.meta.data-source", "fileinmem")
import json

with open("../syn-question-col-analysis/question_generation/gen_output/msmarco/selected_corpus_lm_fcm_STD2_L10000_gpt-neo-1.3B_BS_5_E13931.459746599197.jsonl") as f:
    questions = [line for line in map(json.loads, f)]

questions = questions[:200]
for at in [10, 100, 1000, 10000]:
    bm25_pipe = pt.rewrite.tokenise() >> pt.BatchRetrieve(index, wmodel="BM25", num_results=at)#).parallel(3)
    
    results = []
    time_list = []
    st = time.time()
    
    for question in tqdm(questions):
        
        questions_dataframe = pd.DataFrame([{"qid":0, "query":question["question"].lower()}])

        df_results = bm25_pipe.transform(questions_dataframe)
        
        results.append(df_results["docno"].tolist())
        
    print(at, ":", 200/(time.time()-st),"it/s")
    
