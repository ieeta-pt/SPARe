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
    
    dataset_path = dataset_folder
    
    if os.path.exists(os.path.join(dataset_path, "terrier_index")):
        print(f"Skip {dataset_path}, terrier index already exists")
        return 0
       
    print(dataset_path)
    print()
    #if os.path.exists(os.path.join(dataset_path, "terrier_index")):
    #    print(f"Skipping {dataset_path} since it already exists")
    #    continue
    
    def load_collection(dataset_path):
        filter_duplicates = set()
        print()
        #print(os.listdir(dataset_path, "collection"))
        corpus_file_name = list(filter(lambda x: "corpus" in x, os.listdir(os.path.join(dataset_path, "collection"))))[0]
        
        corpus_file_path = os.path.join(dataset_path, "collection", corpus_file_name)
        
        with open(corpus_file_path) as f:
            for line in f:
                data = json.loads(line)
                if data["id"] not in filter_duplicates and len(data["contents"])>2:
                    yield {"docno": data["id"], "text": data["contents"]}
                    filter_duplicates.add(data["id"])
    
    index_path = os.path.join("./",dataset_path, "terrier_index")
    
    #if os.path.exists(index_path):
    #    shutil.rmtree(index_path)
    
    print(pt.ApplicationSetup.appProperties.getProperty("max.term.length", "none"))
    
    iter_indexer = pt.IterDictIndexer(index_path, meta={'docno': 50, 'text': 8192})
    iter_indexer.index(load_collection(dataset_path))


if __name__=="__main__":

    main()