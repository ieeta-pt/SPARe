
from enum import Enum
from collections import defaultdict
from dataclasses import dataclass

@dataclass
class RetrievalOutput:
    ids: list
    scores: list
    timmings: tuple = None

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractBackend(metaclass=Singleton):
    
    def __init__(self, devices):
        self.devices=devices
    
    def create_zero_tensor(self, shape, dtype):
        raise NotImplementedError
    
    def create_tensor(self, py_data, dtype):
        raise NotImplementedError
    
    def assign_data_to_tensor(self, tensor, indices_slices, values):
        raise NotImplementedError
    
    def create_coo_matrix(self, indices, values, shape):
        raise NotImplementedError
    
    def assign_data_to_tensor(self, tensor, indices_slices, values, dtype):
        raise NotImplementedError
    
    def assign_tensor_to_tensor(self, tensor, indices_slices, values):
        raise NotImplementedError
    
    def create_zero_tensor(self, shape, dtype):
        raise NotImplementedError
    
    def create_coo_matrix(self, indices, values, shape, dtype):
        raise NotImplementedError
    
    def get_slice_from_tensor(self, tensor, indices_slices):
        raise NotImplementedError
    
    def get_value_from_tensor(self, tensor, index):
        raise NotImplementedError
    
    def sum_of_tensor(self, tensor):
        raise NotImplementedError
    
    def save_tensors_to_file(self, tensors, file_name):
        raise NotImplementedError
        
    def load_tensors_from_file(self, file_name):
        raise NotImplementedError
    
    def lookup_by_indices(self, tensor, dim, indices):
        raise NotImplementedError
    
    def create_csr_matrix(self, crow_indices, col_indices, values, dtype):
        raise NotImplementedError
    
    def create_dense_tensor_from_bow(self, bow, vocab_size, dtype):
        raise NotImplementedError
    
    def accelerate(self, tensor):
        raise NotImplementedError
    
    def fused_retrieve(self, questions_list, question_func, collection, top_k, collect_at=5000):
        raise NotImplementedError