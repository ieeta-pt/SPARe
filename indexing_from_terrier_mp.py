import multiprocessing as mp



from collection import SparseCollection, SparseCollectionCSR
from backend import TYPE
import json
from text2vec import BagOfWords
import torch
from collections import defaultdict
from tqdm import tqdm
from weighting_model import BM25Transform
import click
import pyterrier  as pt
import pandas as pd
import glob
import re

from multiprocessing import Queue

def readerworker(q_out: Queue, path_to_msmarco, max_lines):
    
    def get_title_abstract(string):
        data = json.loads(string)
        title, abstract = data["title"], data["abstract"]
        return f"{title} {abstract}"

    with open(path_to_msmarco) as f:
        for i, article_text in enumerate(map(get_title_abstract,f)):
            q_out.put((i, article_text))
            if i>=max_lines:
                break
            
    q_out.put(None)

def tokenizerworker(q_in: Queue, q_out: Queue, msmarco_folder, identifier):
    

    if not pt.started():
        pt.init()
    #print("Init Tokenizer 3")
    indexref = pt.IndexRef.of(f"{msmarco_folder}/terrier_index")
    index = pt.IndexFactory.of(indexref)
    #print("Init Tokenizer 4")
    def tp_func():
        stops = pt.autoclass("org.terrier.terms.Stopwords")(None)
        stemmer = pt.autoclass("org.terrier.terms.PorterStemmer")(None)
        def _apply_func(row):
            words = row["query"].split(" ") # this is safe following pt.rewrite.tokenise()
            words = [stemmer.stem(w) for w in words if not stops.isStopword(w) ]
            return words#" ".join(words)
        return _apply_func 
    #print("Init Tokenizer 5")
    pipe = pt.rewrite.tokenise() >> pt.apply.query(tp_func())
    token2id = {word.getKey():i for i,word in enumerate(index.getLexicon()) }

    vocab_size = len(index.getLexicon())
    #print("Init Tokenizer 6")
    def tokenizer(text):
        tokens_ids = []
        for token in pipe(pd.DataFrame([{"qid":0, "query":text.lower()}]))["query"][0]:
            if token in token2id:
                token_id=token2id[token]
                if token_id is not None:
                    tokens_ids.append(token_id)
        return tokens_ids
    #print("Init Tokenizer 7")
    bow = BagOfWords(tokenizer, vocab_size)
    print("Init Tokenizer running")
    
    while True:
            
        data = q_in.get()
        
        if data is None:
            
            break
        
        doc_id, text = data
        
        #q_in.task_done()
        q_out.put((doc_id, bow(text)))

    #print(f"tok {identifier} found none lets stop" )
    q_out.put(None)
    q_in.put(None)

    
@click.command()
@click.argument("msmarco_folder")
@click.option("--process", default=8)
@click.option("--max_lines", default=-1)
def main(msmarco_folder, process, max_lines):
    
    PATH_TO_MSMARCO = list(glob.glob(f"{msmarco_folder}/corpus*"))[0]
    if not pt.started():
        pt.init()

    indexref = pt.IndexRef.of(f"{msmarco_folder}/terrier_index")
    index = pt.IndexFactory.of(indexref)

    vocab_size = len(index.getLexicon())
    
    if max_lines==-1:
        max_lines = int(re.findall(r"_L([0-9]+)", PATH_TO_MSMARCO)[0])        
    
    manager = mp.Manager()
        
    reader_queue = manager.JoinableQueue(100_000)

    tokenizer_queue = manager.JoinableQueue(100_000)

    reader_process = mp.Process(target=readerworker, 
                                        args=(reader_queue, PATH_TO_MSMARCO, max_lines), 
                                        daemon=True)
    
    tokenizer_processes = [mp.Process(target=tokenizerworker, 
                                        args=(reader_queue, tokenizer_queue, msmarco_folder, i), 
                                        daemon=True) for i in range(process)]
    reader_process.start()
    
    for tok_p in tokenizer_processes:
        tok_p.start()
    
    def get_vectors():
        stop_counter = 0
        while True:
            
            data = tokenizer_queue.get()
            if data is None:
                #print("tok stoped")
                stop_counter += 1
                if stop_counter>=process:
                    break
                continue
            
            doc_id, bow_repr = data
            #tokenizer_queue.task_done()
            yield doc_id, bow_repr

    #exit()
    
    print("start consuming")
    sparseCSR_collection = SparseCollectionCSR.from_vec_iterator(get_vectors(),
                                                                collection_maxsize=max_lines,
                                                                vec_dim=vocab_size,
                                                                dtype=TYPE.float32,
                                                                indices_dtype=TYPE.int32,
                                                                backend="torch") 
    #print("loop_over")
    sparseCSR_collection.save_to_file(f"csr_msmarco")

    sparseCSR_collection.transform(BM25Transform(k1=1.2, b=0.75))

    sparseCSR_collection.save_to_file(f"csr_msmarco_bm25_12_075")
    
    print("reader_process, join")
    reader_process.join()
    
    for i,tok_p in enumerate(tokenizer_processes):
        print(f"tok {i}, join")
        tok_p.join()


if __name__ == '__main__':
    mp.set_start_method("spawn")
    main()