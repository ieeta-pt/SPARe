
from typing import Any
from backend import TYPE
from tqdm import tqdm
from collections import defaultdict
import math

def idf_weighting(x, collection_size):
    #https://en.wikipedia.org/wiki/Okapi_BM25
    return math.log((collection_size-x+0.5)/(x+0.5)+1)

class BM25Transform:
    
    def __init__(self, k1=0.7, b=0.7, idf_weighting=idf_weighting) -> None:
        self.k1 = k1
        self.b = b
        self.idf_weighting=idf_weighting
        
    def __call__(self, *sparce_vecs, metadata, dtype, backend) -> Any:
        self.convert(*sparce_vecs, metadata, dtype, backend)
    
    def convert(self, *args, **kwargs):
        raise RuntimeError
    
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
        
        for doc_id in tqdm(range(crow_indices.shape[0]-1), desc="Converting to BM25 matrix"):
            row_ptr, row_ptr_next = backend.get_slice_from_tensor(crow_indices, slice(doc_id, doc_id+2,None))
            d_constant = d_partial_constant1 + d_partial_constant2*metadata.dl[doc_id]
            
            term_ids = backend.get_slice_from_tensor(col_indices, slice(row_ptr,row_ptr_next,None))
            term_freq =  backend.get_slice_from_tensor(values, slice(row_ptr,row_ptr_next,None))
            
            idf_of_terms = backend.lookup_by_indices(idf_tensor, 0, term_ids)
            
            bm25_term_values = idf_of_terms * ((term_freq*k1_plus_one)/(term_freq+d_constant))
            
            backend.assign_tensor_to_tensor(values, slice(row_ptr,row_ptr_next), bm25_term_values)
