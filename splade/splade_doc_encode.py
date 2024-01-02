import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex
import pandas as pd
import pyt_splade
import torch
import json

dataset = pt.get_dataset('irds:msmarco-passage')

dataset.get_corpus_iter()

splade = pyt_splade.SpladeFactory()

def batch_iterator(iterator, batch_size=16):

    batch = {'text': [], 'docno': []}
    for sample in iterator:
        batch['text'].append(sample['text'])
        batch['docno'].append(sample['docno'])
        if len(batch['text']) == batch_size:
            yield batch
            batch = {'text': [], 'docno': []}
    if len(batch['text'])>0:
        yield batch

with open("splade_msmarco_bow.jsonl", "w") as f:
    
    for doc_sample in batch_iterator(dataset.get_corpus_iter(), batch_size=64):

        with torch.no_grad():
            doc_reps = splade.model(d_kwargs=splade.tokenizer(
                                doc_sample["text"],
                                add_special_tokens=True,
                                padding="longest",  # pad to max sequence length in batch
                                truncation="longest_first",  # truncates to max model length,
                                max_length=splade.max_length,
                                return_attention_mask=True,
                                return_tensors="pt",
                            ).to(splade.device))["d_rep"]
            
        for i, doc_id in enumerate(doc_sample["docno"]):
            cols = torch.nonzero(doc_reps[i]).squeeze().cpu().tolist()
            # and corresponding weights               
            weights = doc_reps[i,cols].cpu().tolist()
            
            d_bow = {splade.reverse_voc[k] : v for k, v in sorted(zip(cols, weights), key=lambda x: (-x[1], x[0]))}
            
            data = {
                "docno": doc_id,
                "bow": d_bow
            }
            
            f.write(f"{json.dumps(data)}\n")