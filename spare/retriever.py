import time 
from tqdm import tqdm
import numpy as np
import math

class SparseRetriever:
    
    def __init__(self, collection, bow, weighting_model=None):
        # apply ranking_model
        if weighting_model is None:
            self.weighting_model = collection.weighting_schema.get_weighting_model()
        else:
            self.weighting_model = weighting_model
        
        self.backend = collection.backend
        self.bow = bow
        
        self.collection = self.weighting_model.transform_collection(collection)
        
        
        # apply ranking_model?

    def _retrieve(self, question_iterator, top_k=1000):
        ## single retrieve? is worth it to do it???
        sparse_collection_matrix_acc = self.backend.accelerate(self.collection.get_sparse_matrix())
        
        #indices = []
        #values = []
        
        for question in tqdm(question_iterator):

            question = self.weighting_model.question_transform(self.bow(question))
            
            # perform search
            query = self.backend.create_dense_tensor_from_bow(question)
            query = self.backend.accelerate(query)
            #out = self.backend.
        
    
    def retrieve(self, questions_list, top_k=1000, collect_at=5000, profiling=False, return_scores=False):
        
        top_k = min(top_k, self.collection.shape[0])
        
        def query_transform(query):
            return self.weighting_model.transform_query(self.bow(query))
        
        # leave 20‰ free for the reminder of the computations
        mem_required = self.collection.get_sparse_matrix_space() * 0.2
        
        if self.backend.num_devices>1:
            # lets take advantage of multi devices
            
            # add (10% TODO find best value for bottlenecks) for bottlenecks
            
            if mem_required < self.backend.get_available_memory_per_device_inGB():
                out = self.backend.dp_retrieval(questions_list, 
                                            query_transform,
                                            self.collection, 
                                            top_k,
                                            collect_at=collect_at,
                                            profiling=profiling,
                                            return_scores=return_scores)
            else:
                raise NotImplementedError("Tensor parallelism is currently not supported in the torch backend!")
            
        else:
            if mem_required < self.backend.get_available_memory_per_device_inGB():
                out = self.backend.forward_retrieval(questions_list, 
                                                query_transform,
                                                self.collection, 
                                                top_k,
                                                collect_at=collect_at,
                                                profiling=profiling,
                                                return_scores=return_scores)
            else:
                
                shards_count = math.ceil(mem_required/self.backend.get_available_memory_per_device_inGB())
                
                out = self.backend.sharding_retrieval(questions_list, 
                                                query_transform,
                                                self.collection, 
                                                top_k,
                                                shards_count=shards_count,
                                                collect_at=collect_at,
                                                profiling=profiling,
                                                return_scores=return_scores)
        
        # maybe convert index2docID if docID!=index
        s_time = time.time()
        converted_indices = []
        for i in range(len(out.ids)):
            q_indices = self.collection.metadata.index2docID[out.ids[i]]
            converted_indices.append(q_indices)
        print("Time to convert docs ids",time.time()-s_time )
        out.ids = np.array(converted_indices)
        
        return out
  
            
            
            