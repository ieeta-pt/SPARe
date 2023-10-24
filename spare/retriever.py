import time 
from tqdm import tqdm
import numpy as np

class SparseRetriever:
    
    def __init__(self, collection, bow, weighting_model=None):
        # apply ranking_model
        if weighting_model is None:
            self.weighting_model = collection.get_weighting_model()
        else:
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
        
    
    def retrieve(self, questions_list, top_k=1000, collect_at=5000, profiling=False, return_scores=False):
        
        def query_transform(query):
            return self.weighting_model.transform_query(self.bow(query)
                                                           )
        out = self.backend.fused_retrieve(questions_list, 
                                           query_transform,
                                           self.collection, 
                                           top_k,
                                           collect_at=collect_at,
                                           profiling=profiling,
                                           return_scores=return_scores)
        
        # maybe convert index2docID if docID!=index
        
        s_time = time.time()
        converted_indices = []
        for i in range(len(out.ids)):
            q_indices = self.collection.metadata.index2docID[out.ids[i]]#[self.collection.metadata.index2docID[idx] for idx in out.ids[i]]
            converted_indices.append(q_indices)
        print("Time to convert docs ids",time.time()-s_time )
        out.ids = np.array(converted_indices)
        
        return out
  
            
            
            