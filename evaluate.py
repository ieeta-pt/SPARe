from ranx import Qrels, Run, evaluate
from collections import defaultdict

def evaluate_spare(qrels, spare_result, question_ids):
    ranx_run = defaultdict(dict)
    
    for i in range(len(spare_result.ids)):
        scores = spare_result.scores[i].tolist()
        for j in range(len(spare_result.ids[i])):
            ranx_run[question_ids[i]][spare_result.ids[i][j]] = scores[j]
            
    qrels = Qrels(qrels)
    ranx_sp_run = Run(ranx_run)
    
    return evaluate(qrels, ranx_sp_run, ["recall@1000", "ndcg@10", "ndcg@10000"])

def evaluate_list(qrels, results, question_ids):
    
    ranx_run = defaultdict(dict)
    
    for i, q_resutls in enumerate(results):
        #print(len(q_resutls), q_resutls)
        for doc_id, doc_score in q_resutls:
            ranx_run[question_ids[i]][doc_id] = doc_score
    
    qrels = Qrels(qrels)
    ranx_sp_run = Run(ranx_run)
    
    return evaluate(qrels, ranx_sp_run, ["recall@1000", "ndcg@10", "ndcg@10000"], make_comparable=True)