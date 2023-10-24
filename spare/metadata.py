from collections import defaultdict
from spare.utils import get_best_np_dtype_for
from enum import Enum
import pickle
import numpy as np

class MetaDataDocID:
    # stores a mapping for the doc_ID
    def __init__(self) -> None:
        self.index2docID = []
        
    def register_docid(self, index, docID):
        assert(index == len(self.index2docID))
        self.index2docID.append(docID)

    def optimize(self):
        # convert to np array
        try:
            self.index2docID = list(map(int, self.index2docID))
            # select the best dtype
            dtype = get_best_np_dtype_for(min(self.index2docID), max(self.index2docID))
            self.index2docID = np.array(self.index2docID, dtype=dtype)
        except ValueError as e:
            # will default to str or object in the worst case
            self.index2docID = np.array(self.index2docID)
        
    def _get_vars_to_save(self):
        return {"index2docID": self.index2docID}
    
    def save_to_file(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self._get_vars_to_save(), f)
    
    def _load_vars(self, data):
        self.index2docID = data.pop("index2docID")
    
    def load_from_file(self, file_name):
        with open(file_name, "rb") as f:
            data = pickle.load(f)
        
        self._load_vars(data)
        
class MetaDataDFandDL(MetaDataDocID):
    # stores the doc freq and doc len as metadata for future usage
    def __init__(self) -> None:
        super().__init__()
        self.df = defaultdict(int)
        self.dl = {}
        
    def update(self, index_doc, term_ids, values):
        self.dl[index_doc] = sum(values)
        for term_id in term_ids:
            self.df[term_id] += 1
    
    def _get_vars_to_save(self):
        return super()._get_vars_to_save() | {
            "df": self.df,
            "dl": self.dl
            } 

    def _load_vars(self, data):
        self.df = data.pop("df")
        self.dl = data.pop("dl")
        super()._load_vars(data)