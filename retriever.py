import time 
from tqdm import tqdm

class SparseRetriever:
    
    def __init__(self, collection, bow, weighting_model):
        # apply ranking_model
        self.weighting_model = weighting_model
        
        self.backend = collection.backend
        self.bow = bow
        
        self.collection = self.weighting_model.transform_collection(collection)
        
        
        # apply ranking_model?

    def _retrieve(self, question_iterator, top_k=1000):
        sparse_collection_matrix_acc = self.backend.accelerate(self.collection.get_sparce_matrix())
        
        #indices = []
        #values = []
        
        for question in tqdm(question_iterator):

            question = self.weighting_model.question_transform(self.bow(question))
            
            # perform search
            query = self.backend.create_dense_tensor_from_bow(question)
            query = self.backend.accelerate(query)
            #out = self.backend.
        
    
    def retrieve(self, questions_list, top_k=1000, collect_at=5000):
        
        def query_transform(query):
            return self.weighting_model.transform_query(self.bow(query)
                                                           )
        return self.backend.fused_retrieve(questions_list, 
                                           query_transform,
                                           self.collection, 
                                           top_k,
                                           collect_at=collect_at)
  
            
            
            