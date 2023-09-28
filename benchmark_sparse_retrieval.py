import click
import pyterrier  as pt
import pandas as pd
from text2vec import BagOfWords
from safetensors import safe_open
import torch
import json
import time
from tqdm import tqdm

@click.command()
@click.argument("dataset_folder")
def main(dataset_folder):
    pt.init()

    indexref = pt.IndexRef.of(f"{dataset_folder}/terrier_index")
    index = pt.IndexFactory.of(indexref)

    def tp_func():
        stops = pt.autoclass("org.terrier.terms.Stopwords")(None)
        stemmer = pt.autoclass("org.terrier.terms.PorterStemmer")(None)
        def _apply_func(row):
            words = row["query"].split(" ") # this is safe following pt.rewrite.tokenise()
            words = [stemmer.stem(w) for w in words if not stops.isStopword(w) ]
            return words#" ".join(words)
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
    
    tensors = {}
    with safe_open(f"csr_msmarco_bm25_pyterrier.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
        
    tensors = [tensors["vec_0"], tensors["vec_1"], tensors["vec_2"]]

    csr_matrix_cpu = torch.sparse_csr_tensor(*tensors, (8841823, vocab_size))
    
    csr_matrix_gpu = csr_matrix_cpu.to("cuda")
    
    with open(f"{dataset_folder}/selected_corpus_lm_fcm_STD2_L10000_gpt-neo-1.3B_BS_5_E13931.459746599197.jsonl") as f:
    #with open("../syn-question-col-analysis/question_generation/gen_output/msmarco/selected_corpus_lm_fcm_STD2_L10000_gpt-neo-1.3B_BS_5_E13931.459746599197.jsonl") as f:
        questions = [line for line in map(json.loads, f)]
    
    def text_to_dense_torch(text):
        b = bow(text)
        return torch.sparse_coo_tensor([list(b.keys())], list(b.values()), (vocab_size,), dtype=torch.float32).to_dense()

    def retrieve_topk_gpu(question, at=10):
        query_gpu = text_to_dense_torch(question).to("cuda")
        r = torch.topk(csr_matrix_gpu @ query_gpu, k=at, dim=0).indices#.cpu()
        #print("tokenizer stack", end_stack-end_tok_t, "retrieve",end_r_t-end_stack)
        return r
    
    print("start search")

    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA:", torch.version.cuda)
    print("Memory Usage:")
    print("Allocated:", round(torch.cuda.memory_allocated(0)/1024**3,2),"GB")

    for at in [10, 100, 1000, 10000]:

        results = []
        total_time = 0
        total_time_gpu = 0
        for question in questions[:1000]:
            start_bow = time.time()
            b = bow(question["question"])
            start_t = time.time()
            query_gpu = torch.sparse_coo_tensor([list(b.keys())], list(b.values()), (vocab_size,), dtype=torch.float32).to_dense().to("cuda")
            r = torch.topk(csr_matrix_gpu @ query_gpu, k=at, dim=0).indices
            results.append(r)
            end_t = time.time()
            total_time_gpu+=(end_t - start_t)
            total_time+=(end_t - start_bow)

        print(f"at: {at}", 1000/total_time, "| only GPU", 1000/total_time_gpu)

    

if __name__ == '__main__':
    main()