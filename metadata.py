from collections import defaultdict
import json

class MetaDataDFandDL:
    # stores the doc freq and doc len as metadata for future usage
    def __init__(self) -> None:
        self.df = defaultdict(int)
        self.dl = {}
        
    def update(self, doc_id, term_ids, values):
        self.dl[doc_id] = sum(values)
        for term_id in term_ids:
            self.df[term_id] += 1
            
    def save_to_file(self, file_name):
        with open(file_name, "w") as f:
            json.dump({
                "df": self.df,
                "dl": self.dl
                }, f)
            
    def load_from_file(self, file_name):
        with open(file_name) as f:
            data = json.load(f)
        
        self.df = data["df"]
        self.dl = data["dl"]
            