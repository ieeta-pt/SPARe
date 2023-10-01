from collection import SparseCollection, SparseCollectionCSR
from backend import TYPE
import json
import psutil
from text2vec import BagOfWords
import torch
from collections import defaultdict
from tqdm import tqdm
import multiprocessing  as mp

import pyterrier  as pt
import pandas as pd

import torch.nn as nn

class SparseRetrievalModel(nn.Module):
    def __init__(self, sparse_collection, top_k = 10):
        super().__init__()
        #self.shape = sparse_collection.sparse_vecs, sparse_collection.shape
        self.crow = torch.nn.parameter.Parameter(sparse_collection.sparse_vecs[0], requires_grad=False)
        self.indice = torch.nn.parameter.Parameter(sparse_collection.sparse_vecs[1], requires_grad=False)
        self.values = torch.nn.parameter.Parameter(sparse_collection.sparse_vecs[2], requires_grad=False)
        self.collection_matrix = None#torch.sparse_csr_tensor(self.crow, self.indice, self.values, sparse_collection.shape)
        self.shape = sparse_collection.shape
        self.top_k = top_k
        
    def forward(self, x):
        x= x.squeeze(0)
        #print(x.shape)
        collection_matrix = torch.sparse_csr_tensor(self.crow, self.indice, self.values, self.shape)
    
        return torch.topk(collection_matrix @ x, k=self.top_k, dim=0)
        #return x


class QDS(torch.utils.data.Dataset):
    def __init__(self, questions, bow):
        self.questions = questions#[:10000]
        self.bow = bow

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        b = self.bow(self.questions[idx]["question"])
        return torch.sparse_coo_tensor([list(b.keys())], list(b.values()), (vocab_size,), dtype=torch.float32).to_dense()
pt.init()

indexref = pt.IndexRef.of("../syn-question-col-analysis/datasets/msmarco/terrier_index/")
index = pt.IndexFactory.of(indexref)

def tp_func():
    stops = pt.autoclass("org.terrier.terms.Stopwords")(None)
    stemmer = pt.autoclass("org.terrier.terms.PorterStemmer")(None)
    def _apply_func(row):
        words = row["query"].split(" ") # this is safe following pt.rewrite.tokenise()
        words = [stemmer.stem(w) for w in words if not stops.isStopword(w) ]
        return words
    return _apply_func 

pipe = pt.rewrite.tokenise() >> pt.apply.query(tp_func())
token2id = {word.getKey():i for i,word in enumerate(index.getLexicon()) }

vocab_size = len(index.getLexicon())

def tokenizer(text):
    tokens_ids = []
    for token in pipe(pd.DataFrame([{"qid":0, "query":text.lower()}]))["query"][0]:
        if token in token2id:
            token_id=token2id[token]
            if token_id is not None:
                tokens_ids.append(token_id)
    return tokens_ids

bow = BagOfWords(tokenizer, vocab_size)


if __name__=="__main__":
    #mp.set_start_method("spawn")
    

    sparse_collection = SparseCollectionCSR.load_from_file("csr_msmarco_bm25_12_075_terrier")

    
    retrieval_model = SparseRetrievalModel(sparse_collection, 1000)

    with open("../syn-question-col-analysis/question_generation/gen_output/msmarco/selected_corpus_lm_fcm_STD2_L10000_gpt-neo-1.3B_BS_5_E13931.459746599197.jsonl") as f:
        questions = [line for line in map(json.loads, f)]
        
    def text_to_dense_torch(text):
        b = bow(text)
        return torch.sparse_coo_tensor([list(b.keys())], list(b.values()), (vocab_size,), dtype=torch.float32).to_dense()

    def text_to_sparse_torch(text):
        b = bow(text)
        return torch.sparse_coo_tensor([list(b.keys())], list(b.values()), (vocab_size,), dtype=torch.float32)


    retrieval_model_gpu = retrieval_model.to("cuda")

    replicas = nn.parallel.replicate(retrieval_model_gpu, [0,1])

    print("start search")

    #print("GPU:", torch.cuda.get_device_name(0))
    #print("CUDA:", torch.version.cuda)
    #print("Memory Usage:")
    #print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,2),"GB")

    devices = list(range(torch.cuda.device_count()))

    dataloader = torch.utils.data.DataLoader(QDS(questions, bow), 
                                            batch_size=len(devices), 
                                            pin_memory=True, 
                                            num_workers=1)
    import time
    results = {i:{"indices":[],"values":[]} for i in [0,1]}
    #results = []
    total_time = 0
    total_time_gpu = 0
    query_batch = []
    for questions in tqdm(dataloader):
        inputs = nn.parallel.scatter(questions, devices)        
        r = nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)
    #results.append(r)
        for i, out in enumerate(r):
            results[i]["indices"].append(out.indices)
            results[i]["values"].append(out.values)

        #results.extend([out.indices for out in r])
        #r = retrieval_model_multigpu(inputs).indices
        #print(r.shape)
        

    indices_cpu = []
    values_cpu = []
    s = time.time()
    for d_id, out in results.items():
        indices_cpu.append(torch.stack(out["indices"]).cpu())
        values_cpu.append(torch.stack(out["values"]).cpu())
    e = time.time()
    print("transfer to cpu", e-s)