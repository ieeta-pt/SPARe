from utils import idf_weighting
from enum import Enum
import json
from typing import Any
from backend import TYPE
from tqdm import tqdm
from utils import idf_weighting
from metadata import MetaDataDFandDL
import jsonpickle

class WeightingSchemaType(Enum):
    counting = 1
    bm25 = 2

class WeightingSchema:

    def __init__(self, type) -> None:
        self.type = type
        
    def _get_vars_to_save(self):
        return {"type": self.type}
    
    def save_to_file(self, path_to_file):
        encoded_vars = jsonpickle.encode(self._get_vars_to_save())
        with open(path_to_file, "w") as f:
            f.write(encoded_vars)

    def _load_vars(self, data):
        self.type = data.pop("type")
    
    def load_from_file(self, path_to_file):
        with open(path_to_file, "r") as f:
            data = jsonpickle.decode(f.read())
        
        self._load_vars(data)
        
class CountingWeightingSchema(WeightingSchema):
    def __init__(self) -> None:
        super().__init__(WeightingSchemaType.counting)
        
    def __repr__(self) -> str:
        return "CountingWeightingSchema"
        
class BM25WeightingSchema(WeightingSchema):
    def __init__(self, k1=None, b=None, idf_weighting=None) -> None:
        super().__init__(WeightingSchemaType.bm25)
        self.k1 = k1
        self.b = b
        self.idf_weighting = idf_weighting
    
    def _get_vars_to_save(self):
        return super()._get_vars_to_save() | {
            "k1": self.k1,
            "b": self.b,
            "idf_weighting": idf_weighting
            } 
    
    def _load_vars(self, data):
        self.k1 = data.pop("k1")
        self.b = data.pop("b")
        self.idf_weighting = data.pop("idf_weighting")
        super()._load_vars(data)
        
    def __repr__(self) -> str:
        return f"BM25WeightingSchema(k1:{self.k1}, b:{self.b}, idf: {self.idf_weighting})"

class BM25WeightingModel:
    
    def __init__(self, k1=None, b=None, k3=None, idf_weighting=None) -> None:
        self.k1 = k1
        self.b = b
        self.k3 = k3
        self.idf_weighting=idf_weighting
    
    def transform_query(self, query):
        # apply the weighting scheme to the query bow if its applicable
        return query
    
    def transform_collection(self, collection, inplace=True):
        
        if collection.weighting_schema.type==WeightingSchemaType.bm25:
            # its already in the correct weightingSchema
            self.k1 = collection.weighting_schema.k1
            self.b = collection.weighting_schema.b
            self.idf_weighting = collection.weighting_schema.idf_weighting
            print("Collection is already in BM25 weighting schema, using its parameters")
            return collection
        
        if not inplace:
            collection = collection.copy()
        
        collection.transform(BM25Transform(k1=self.k1, b=self.b, idf_weighting=self.idf_weighting))
        return collection

class BM25Transform:
    
    def __init__(self, k1=0.7, b=0.7, idf_weighting=idf_weighting) -> None:
        self.k1 = k1
        self.b = b
        self.idf_weighting=idf_weighting
        
    def __call__(self, *sparce_vecs, metadata, dtype, backend) -> Any:
        self.convert(*sparce_vecs, metadata, dtype, backend)
    
    def convert(self, *args, **kwargs):
        raise RuntimeError
    
    def get_weighting_schema(self):
        return BM25WeightingSchema(k1=self.k1, b=self.b, idf_weighting=self.idf_weighting)
    
    def is_compatible(self, collection):
        # bm25 can only be applied to collection with weigting scheme that is of type counting
        # and metadata that hold dl and df metadata
        return collection.weighting_schema.type == WeightingSchemaType.counting \
            and isinstance(collection.metadata, MetaDataDFandDL)
    
    def for_coo(self):
        return BM25TransformForCOO(k1=self.k1, b=self.b, idf_weighting=self.idf_weighting)
    
    def for_csr(self):
        return BM25TransformForCSR(k1=self.k1, b=self.b, idf_weighting=self.idf_weighting)
    
class BM25TransformForCOO(BM25Transform):
    
    def _precomputations(self, metadata):
        collection_size = len(metadata.dl)
        avgdl = sum(metadata.dl.values())/len(metadata.dl)
        idf = {term_id: self.idf_weighting(cnt, collection_size) for term_id, cnt in metadata.df.items()}
        
        k1_plus_one = self.k1+1
        k1_times_b = self.k1*self.b
        d_partial_constant1 = self.k1 - k1_times_b
        d_partial_constant2 = k1_times_b/avgdl
        
        return k1_plus_one, d_partial_constant1, d_partial_constant2, idf
        
    def convert(self, indices, values, shape, nnz, metadata, dtype, backend):
        
        k1_plus_one, d_partial_constant1, d_partial_constant2, idf = self._precomputations(metadata)
        
        for i in tqdm(range(nnz), desc="Converting to BM25 matrix"):
            doc_id = backend.get_value_from_tensor(indices, (0,i))
            term_id = backend.get_value_from_tensor(indices, (1,i))
            value = backend.get_value_from_tensor(values, i)
            
            bm25_w_value = idf[term_id] * ((value * k1_plus_one) / (value + d_partial_constant1 + d_partial_constant2*metadata.dl[doc_id]))
            
            backend.assign_data_to_tensor(values, i, bm25_w_value, dtype)
         
        
class BM25TransformForCSR(BM25TransformForCOO):
    
    def convert(self, crow_indices, col_indices, values, shape, nnz, metadata, dtype, backend):
        
        k1_plus_one, d_partial_constant1, d_partial_constant2, idf = self._precomputations(metadata)
        
        #idf to dense tensor
        idf_tensor = backend.create_coo_matrix([list(idf.keys())], list(idf.values()), (shape[1],), dtype=TYPE.float32).to_dense()
        
        for doc_id in tqdm(range(crow_indices.shape[0]-1), desc="Converting to BM25 weighted collection"):
            row_ptr, row_ptr_next = backend.get_slice_from_tensor(crow_indices, slice(doc_id, doc_id+2,None))
            d_constant = d_partial_constant1 + d_partial_constant2*metadata.dl[doc_id]
            
            term_ids = backend.get_slice_from_tensor(col_indices, slice(row_ptr,row_ptr_next,None))
            term_freq =  backend.get_slice_from_tensor(values, slice(row_ptr,row_ptr_next,None))
            
            idf_of_terms = backend.lookup_by_indices(idf_tensor, 0, term_ids)
            
            bm25_term_values = idf_of_terms * ((term_freq*k1_plus_one)/(term_freq+d_constant))
            
            backend.assign_tensor_to_tensor(values, slice(row_ptr,row_ptr_next), bm25_term_values)
