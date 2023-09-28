from collection import SparseCollection, get_matrix_estimations
from utils import get_


class SparseRetriever:
    
    def __init__(self, sparse_collection: SparseCollection):
        self.sparse_collection = sparse_collection
        
    