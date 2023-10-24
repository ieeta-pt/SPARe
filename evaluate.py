from ranx import Qrels, Run, evaluate
from collections import defaultdict

def evaluate_spare(qrels, spare_result, question_ids):
    ranx_run = defaultdict(dict)
    spare_result_ids = spare_result.ids#.tolist()
    spare_result_scores = spare_result.scores#.tolist()
    
    # lets truncate to 1000 since the metrics are related to 1k
    max_at = 1000 
    
    print("Converting to ranx format")
    for i in range(len(spare_result_ids)):
        scores = spare_result_scores[i]
        for j in range(min(len(spare_result_scores[0]),max_at)):
            ranx_run[question_ids[i]][str(spare_result_ids[i][j])] = scores[j]
    
    
    print("Build Qrels and Run")
    qrels = Qrels(qrels)
    ranx_sp_run = Run(ranx_run)
    
    print("START EVALUATE?")
    
    return evaluate(qrels, ranx_sp_run, [f"recall@{max_at}", "ndcg@10", f"ndcg@{max_at}"])

def evaluate_list(qrels, results, question_ids):
    
    ranx_run = defaultdict(dict)
    
    for i, q_resutls in enumerate(results):
        #print(len(q_resutls), q_resutls)
        for doc_id, doc_score in q_resutls:
            ranx_run[question_ids[i]][doc_id] = doc_score
    
    qrels = Qrels(qrels)
    ranx_sp_run = Run(ranx_run)
    
    return evaluate(qrels, ranx_sp_run, ["recall@1000", "ndcg@10", "ndcg@1000"], make_comparable=True)