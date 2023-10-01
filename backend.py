
from enum import Enum
from collections import defaultdict
import torch
from safetensors.torch import save_file
from safetensors import safe_open
from tqdm import tqdm


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
    
    def accelerate(self, tensor):
        return tensor.to("cuda:0")
    
    def fused_retrieve(self, questions_list, question_func, collection, top_k):
        
        devices = list(range(torch.cuda.device_count()))
        sparse_model= SparseRetrievalModel(collection, top_k=top_k).to("cuda")
        questions_dataset = QuestionDataset(questions_list, question_func, collection.shape[-1])
        
        dl = torch.utils.data.DataLoader(questions_dataset, 
                                            batch_size=len(devices), 
                                            pin_memory=True, 
                                            num_workers=1)
        
        if len(devices)>1:
            
            replicas = torch.nn.parallel.replicate(sparse_model, devices)
            results = {i:{"indices":[],"values":[]} for i in devices}

            for questions in tqdm(dl):
                
                inputs = torch.nn.parallel.scatter(questions, devices)        
                r = torch.nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)
            #results.append(r)
                for i, out in enumerate(r):
                    results[i]["indices"].append(out.indices)
                    results[i]["values"].append(out.values)
            
            indices_cpu = []
            values_cpu = []

            for d_id, out in results.items():
                indices_cpu.append(torch.stack(out["indices"]).cpu())
                values_cpu.append(torch.stack(out["values"]).cpu())

        else:
            indices = []
            values = []
            for question in tqdm(dl):
                r = sparse_model(question.to("cuda"))
                indices.append(r.indices)
                values.append(r.values)
                
            indices_cpu = torch.stack(indices).cpu()
            values_cpu = torch.stack(indices).cpu()
            
        return indices_cpu, values_cpu
            
class SparseRetrievalModel(torch.nn.Module):
    def __init__(self, sparse_collection, top_k = 10):
        super().__init__()
        #self.shape = sparse_collection.sparse_vecs, sparse_collection.shape
        self.crow = torch.nn.parameter.Parameter(sparse_collection.sparse_vecs[0], requires_grad=False)
        self.indice = torch.nn.parameter.Parameter(sparse_collection.sparse_vecs[1], requires_grad=False)
        self.values = torch.nn.parameter.Parameter(sparse_collection.sparse_vecs[2], requires_grad=False)
        self.collection_matrix = None#torch.sparse_csr_tensor(self.crow, self.indice, self.values, sparse_collection.shape)
        self.shape = sparse_collection.shape
        self.top_k = top_k
        
    def forward(self, x):
        x= x.squeeze(0)
        #print(x.shape)
        collection_matrix = torch.sparse_csr_tensor(self.crow, self.indice, self.values, self.shape)
    
        return torch.topk(collection_matrix @ x, k=self.top_k, dim=0)
        #return x


class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, questions, bow, vocab_size):
        self.questions = questions#[:10000]
        self.bow = bow
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        b = self.bow(self.questions[idx])
        return torch.sparse_coo_tensor([list(b.keys())], list(b.values()), (self.vocab_size,), dtype=torch.float32).to_dense()