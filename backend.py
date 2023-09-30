
from enum import Enum
from collections import defaultdict
import torch
from safetensors.torch import save_file
from safetensors import safe_open

class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
   
class TYPE(Enum):
    int32 = 1
    int64 = 2
    float16 = 3
    float32 = 4

class AbstractBackend(metaclass=Singleton):
    
    def create_zero_tensor(self, shape, dtype):
        raise NotImplementedError
    
    def create_tensor(self, py_data, dtype):
        raise NotImplementedError
    
    def assign_data_to_tensor(self, tensor, indices_slices, values):
        raise NotImplementedError
    
    def create_coo_matrix(self, indices, values, shape):
        raise NotImplementedError
    
class TorchBackend(AbstractBackend):
    
    def __init__(self):
        self.types_converter = {
            TYPE.int32: torch.int32,
            TYPE.int64: torch.int64,
            TYPE.float32: torch.float32,
            TYPE.float16: torch.float16,
        }
    
    def assign_data_to_tensor(self, tensor, indices_slices, values, dtype):
        tensor[indices_slices] = torch.tensor(values, dtype=self.types_converter[dtype])
    
    def assign_tensor_to_tensor(self, tensor, indices_slices, values):
        tensor[indices_slices] = values
    
    def create_zero_tensor(self, shape, dtype):
        return torch.zeros(shape, dtype=self.types_converter[dtype])
    
    def create_coo_matrix(self, indices, values, shape, dtype):
        return torch.sparse_coo_tensor(indices, values, shape, dtype=self.types_converter[dtype])
    
    def get_slice_from_tensor(self, tensor, indices_slices):
        return tensor[indices_slices]
    
    def get_value_from_tensor(self, tensor, index):
        return tensor[index].item()
    
    def sum_of_tensor(self, tensor):
        return tensor.sum()
    
    def save_tensors_to_file(self, tensors, file_name):
        tensor_data = {f"vec_{i}":tensor for i, tensor in enumerate(tensors)}
        save_file(tensor_data, file_name)
        
    def load_tensors_from_file(self, file_name):
        
        tensors = []
        
        with safe_open(file_name, framework="pt", device="cpu") as f:
            for key in sorted(f.keys()):
                tensors.append(f.get_tensor(key))
            
        return tensors
    
    def lookup_by_indices(self, tensor, dim, indices):
        return torch.index_select(tensor, dim, indices)
    
    def create_csr_matrix(self, crow_indices, col_indices, values, dtype):
        return torch.sparse_csr_tensor(crow_indices, col_indices, values, self.types_converter[dtype])
    
    def create_dense_tensor_from_bow(self, bow, vocab_size, dtype):
        return torch.sparse_coo_tensor([list(bow.keys())], list(bow.values()), (vocab_size,), dtype=self.types_converter[dtype]).to_dense()   
    