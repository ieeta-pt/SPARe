from beir import util
from beir.datasets.data_loader import GenericDataLoader
import os
import json
import shutil
import click

#"msmarco",
#"nq",
#"hotpotqa",
#"climate-fever",
#"arguana",
#"quora",
#"scidocs",
#"fever",

datasets = ["arguana", 
            #"climate-fever",
            #"cqadupstack",
            #"dbpedia-entity",
            #"fever",
            #"fiqa",
            #"germanquad",
            "hotpotqa",
            "msmarco",
            #"nfcorpus",
            "nq",
            "quora",
            "scidocs",
            #"scifact",
            #"trec-covid",
            #"vihealthqa",
            #"webis-touche2020"
            ]

def build_corpus_and_rels(data_path, dev_split, dataset_path):
    print(dataset_path, dev_split)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=dev_split)

    # copy corpus
    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    with open(os.path.join(dataset_path,f"corpus_L{len(corpus)}.jsonl"), "w") as fOut:
        for _id, data in corpus.items():
            data_to_write = {
                "pmid": _id,
                "title": data["title"],
                "abstract": data["text"]
            }
            fOut.write(f"{json.dumps(data_to_write)}\n")
            
    with open(os.path.join(dataset_path, "relevant_pairs.jsonl"), "w") as fOut:
        for qid, qtext in queries.items():
            for docid, rel in qrels[qid].items():
                if rel>0 and docid in corpus:
                    
                    data_to_write = {
                        "id": qid,
                        "pmid": docid,
                        "question": qtext,
                        "title": corpus[docid]["title"],
                        "abstract": corpus[docid]["text"]
                    }
                    fOut.write(f"{json.dumps(data_to_write)}\n")
                    
    

    
@click.command()
@click.argument("dataset_folder")
def main(dataset_folder):
    
    for dataset in datasets:
        
        if os.path.exists(os.path.join(dataset_folder, dataset)):
            print(f"Skipping {dataset} since it already exists in {dataset_folder}")
            continue
        
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
        data_path = util.download_and_unzip(url, os.path.join(dataset_folder, ".beir"))
        
        if dataset=="msmarco":
            build_corpus_and_rels(data_path, "dev",  os.path.join(dataset_folder, dataset))
            #build_corpus_and_rels(data_path, "test", "TREC-DL-2019")
        else:
            build_corpus_and_rels(data_path, "test",  os.path.join(dataset_folder, dataset))
            
        shutil.rmtree(os.path.join(dataset_folder, ".beir"))
                    
if __name__ == '__main__':
    main()