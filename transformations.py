
class DtypeTransform:
    
    def __init__(self, dtype) -> None:
        self.dtype = dtype
        
    def __call__(self, collection):
        self.convert(collection)
    
    def convert(self, collection):
        # change the signature to collection it is much better
        values = collection.sparse_vecs[-1]
        values = collection.backend.convert_dtype(values, dtype=self.dtype)
        
        collection.sparse_vecs = collection.sparse_vecs[:-1] + [values]

    def is_compatible(self, collection):
        
        # we should be always able to make values dtype conversion
        return True
    
    def for_coo(self):
        # the logic is the same for coo and csr
        return self
    
    def for_csr(self):
        # the logic is the same for coo and csr
        return self
    
