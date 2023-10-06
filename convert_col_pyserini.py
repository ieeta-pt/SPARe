
import json
from tqdm import tqdm
import click
with open("pyserini_collection/collection.jsonl", "w") as fOut:
    with open("corpus_L8841823.jsonl") as f:
        for data in tqdm(map(json.loads, f)):
            out = {"id": data["pmid"], "contents": data["title"]+" "+ data["abstract"]}
            
            fOut.write(f"{json.dumps(out)}")
            
        