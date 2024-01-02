import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
import pandas as pd
import pyt_splade
import torch
import json
from collections import defaultdict

splade = pyt_splade.SpladeFactory()

def batch_iterator(iterator, batch_size=16):

    batch = defaultdict(list)#{'text': [], 'docno': []}
    for sample in iterator:

        for k in sample.keys():
            batch[k].append(sample[k])
        if len(batch[k]) == batch_size:
            yield batch
            batch = defaultdict(list)
    if len(batch[k])>0:
        yield batch


def query_iterator():
    run = {}
    with open(f"beir_datasets/msmarco/relevant_pairs.jsonl") as f:
        for i, q_data in enumerate(map(json.loads, f)):
            run[q_data["id"]] = q_data["question"]
    

    for q_id, q_text in run.items():
        yield {"id": q_id,
                "question": q_text}
        
with open("splade_msmarco_questions_bow.jsonl", "w") as f:
    
    for query_sample in batch_iterator(query_iterator(), batch_size=64):

        with torch.no_grad():
            query_reps = query_reps = splade.model(q_kwargs=splade.tokenizer(
                        query_sample["question"],
                        add_special_tokens=True,
                        padding="longest",  # pad to max sequence length in batch
                        truncation="longest_first",  # truncates to max model length,
                        max_length=splade.max_length,
                        return_attention_mask=True,
                        return_tensors="pt",
                    ).to(splade.device))["q_rep"]
            
        for i, q_id in enumerate(query_sample["id"]):
            cols = torch.nonzero(query_reps[i]).squeeze().cpu().tolist()
            # and corresponding weights               
            weights = query_reps[i,cols].cpu().tolist()
            
            q_bow = {splade.reverse_voc[k] : v for k, v in sorted(zip(cols, weights), key=lambda x: (-x[1], x[0]))}
            
            data = {
                "docno": q_id,
                "bow": q_bow
            }
            
            f.write(f"{json.dumps(data)}\n")