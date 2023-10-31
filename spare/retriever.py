import time 
from tqdm import tqdm
import numpy as np
import math
from dataclasses import dataclass
from spare import type_to_str
from typing import Callable, Tuple

@dataclass
class Plan:
    pre_init_modules :Tuple
    running_func: Callable
    running_mode_str :str
    algorithm :str


def build_running_plan(backend, collection, objective="performance", algorithm="dot"):
    """
    function responsible to find the best possible configuration to run the 
    retrieval on the available hardware
    """
    # leave at least 20% free
    available_memory_per_device = backend.get_available_memory_per_device_inGB()
    mem_required = collection.get_sparse_matrix_space() * 1.2
    #print(f"Memory required to for the retrieval: {mem_required:.3f} GB", )
    
    # TODO based on the results choose the best combination, probably for collection with more than 2M docs
    
    if backend.num_devices>1:
        if mem_required < available_memory_per_device:
            running_mode_str = "DataParallel"
            if algorithm=="dot":
                algorithm_str = "dot product"
                pre_init_modules = backend.preprare_for_dp_retrieval(collection)
                running_func = backend.dp_retrieval
            else:
                raise NotImplementedError("DataParallel is currently only implemented for dot algorithm")
        else:
            raise NotImplementedError("Tensor parallelism is currently not supported in SPARe")
        
    else:
        if mem_required < available_memory_per_device:
            running_mode_str = "Single forward"
            if algorithm=="dot":
                algorithm_str = "dot product"
                pre_init_modules = backend.preprare_for_forward_retrieval(collection)
                running_func = backend.forward_retrieval
            elif algorithm=="iterative":
                algorithm_str = "iterative product"
                pre_init_modules = backend.preprare_for_forward_retrieval_iterative(collection, objective)
                running_func = backend.forward_retrieval
            else:
                raise NotImplementedError("Single forward is currently only implemented for dot and iterative algorithms")
        else:
            shards_count = math.ceil(mem_required/backend.get_available_memory_per_device_inGB())
            
            running_mode_str = "Sharding"
            algorithm = "dot product"
            pre_init_modules = backend.preprare_for_sharding_retrieval(collection, shards_count)
            running_func = backend.sharding_retrieval

    # print current config
    print()
    print("Runner configuration:")
    print()
    print("Hardware")
    print(f"  accelerators: {backend.devices}")
    print(f"  memory per device: {available_memory_per_device:.2f}")
    print("Collection")
    print(f"  shape: {collection.shape}")
    print(f"  values dtype: {type_to_str(collection.dtype)}")
    print(f"  indices dtype: {type_to_str(collection.indices_dtype)}")
    print(f"  memory required: {collection.get_sparse_matrix_space():.2f}")
    print(f"  memory required (safe margin): {mem_required:.2f}")
    print("Plan")
    print(f"  running mode: {running_mode_str}")
    print(f"  algorithm: {algorithm_str}")
    print(f"  objective: {objective}")
    print()
    
    return Plan(pre_init_modules=pre_init_modules, 
                running_func=running_func, 
                running_mode_str=running_mode_str,
                algorithm=algorithm_str)
    
class SparseRetriever:
    
    def __init__(self, collection, bow, weighting_model=None, objective="accuracy", algorithm="dot"):
        # apply ranking_model
        if weighting_model is None:
            self.weighting_model = collection.weighting_schema.get_weighting_model()
        else:
            self.weighting_model = weighting_model
        
        self.backend = collection.backend
        self.bow = bow
        
        self.collection = self.weighting_model.transform_collection(collection)
        
        # given the available resources, collection stats and objective, pick the best running mode!
        self.running_plan = build_running_plan(self.backend, self.collection, objective=objective, algorithm=algorithm)
        
    def retrieve(self, questions_list, top_k=1000, collect_at=5000, profiling=False, return_scores=False):
        
        top_k = min(top_k, self.collection.shape[0])
        
        def query_transform(query):
            return self.weighting_model.transform_query(self.bow(query))
        
        running_function = self.running_plan.running_func
        
        out = running_function(*self.running_plan.pre_init_modules, 
                               questions_list=questions_list, 
                               question_func=query_transform,
                               top_k=top_k,
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
  
            
            
            