import os
import pyterrier as pt
import json
import shutil
import click

if not pt.started():
    pt.init()

pt.set_property("lowercase","true")
pt.set_property("max.term.length", "150")

@click.command()
@click.argument("dataset_folder")
def main(dataset_folder):
    
    
    
    for dataset in sorted(os.listdir(dataset_folder)):

        dataset_path = os.path.join(dataset_folder, dataset)
        print(dataset_path)
        print()
        if os.path.exists(os.path.join(dataset_path, "terrier_index")):
            print(f"Skipping {dataset_path} since it already exists")
            continue
        
        def load_collection(dataset_path):
            filter_duplicates = set()
            
            corpus_file_name = list(filter(lambda x: "corpus" in x, os.listdir(dataset_path)))[0]
            
            corpus_file_path = os.path.join(dataset_path, corpus_file_name)
            
            with open(corpus_file_path) as f:
                for line in f:
                    data = json.loads(line)
                    if data["pmid"] not in filter_duplicates and len(data["abstract"])>0:
                        yield {"docno": data["pmid"], "text": data["title"]+ " " +data["abstract"]}
                        filter_duplicates.add(data["pmid"])
        
        index_path = os.path.join("./",dataset_path, "terrier_index")
        
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        
        print(pt.ApplicationSetup.appProperties.getProperty("max.term.length", "none"))
        
        iter_indexer = pt.IterDictIndexer(index_path, meta={'docno': 50, 'text': 8192})
        iter_indexer.index(load_collection(dataset_path))


if __name__=="__main__":
    
    main()