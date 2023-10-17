from spare import TYPE
from spare.backend import AbstractBackend, RetrievalOutput
import torch
from safetensors.torch import save_file
from safetensors import safe_open
from tqdm import tqdm
import time

class TorchBackend(AbstractBackend):
    
    def __init__(self):
        if torch.cuda.is_available():
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        else:
            devices = ["cpu"]
        
        super().__init__(devices)
        
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
    
    def create_csr_matrix(self, crow_indices, col_indices, values, shape, dtype):
        return torch.sparse_csr_tensor(crow_indices, col_indices, values, shape, dtype=self.types_converter[dtype])
    
    def create_dense_tensor_from_bow(self, bow, vocab_size, dtype):
        return torch.sparse_coo_tensor([list(bow.keys())], list(bow.values()), (vocab_size,), dtype=self.types_converter[dtype]).to_dense()   
    
    def accelerate(self, tensor):
        return tensor.to(self.devices[0])
    
    def convert_dtype(self, tensor, dtype):
        return tensor.type(self.types_converter[dtype])
    
    def fused_retrieve(self, questions_list, question_func, collection, top_k, collect_at=5000, profiling=False, return_scores=False):
        
        top_k = min(top_k, collection.shape[0])
        if len(self.devices)>1:
            out = self._distributed_retrieval(questions_list, question_func, collection, top_k, collect_at=collect_at, profiling=profiling, return_scores=return_scores)
        else:
            out = self._single_retrieval(questions_list, question_func, collection, top_k, collect_at=collect_at, profiling=profiling, return_scores=return_scores)
                
        converted_indices = []
        for i in range(len(out.ids)):
            q_indices = [collection.metadata.index2docID[idx] for idx in out.ids[i]]
            converted_indices.append(q_indices)
        
        out.ids = converted_indices
        return out

    
    
    def _distributed_retrieval(self, questions_list, question_func, collection, top_k, collect_at, profiling, return_scores):
        
        sparse_model= CSRSparseRetrievalDistributedModel(collection, top_k=top_k).to(self.devices[0])
        questions_dataset = DistributedQuestionDataset(questions_list, question_func, collection.shape[-1])
        
        dl = torch.utils.data.DataLoader(questions_dataset, 
                                         batch_size=len(self.devices), 
                                         collate_fn=distributed_collate_fn, 
                                         pin_memory=True, 
                                         num_workers=0)
        
        replicas = torch.nn.parallel.replicate(sparse_model, self.devices)
        results = {i:{"indices":[],"values":[]} for i,_ in enumerate(self.devices)}
                
        start_retrieval_time = time.time()
        with tqdm(total=len(questions_dataset)) as pbar:
            for questions in dl:

                inputs = torch.nn.parallel.scatter(questions, self.devices)
                r = torch.nn.parallel.parallel_apply(replicas[:len(inputs)], inputs)
            #results.append(r)
                for i, out in enumerate(r):
                    results[i]["indices"].append(out.indices)
                    if return_scores:
                        results[i]["values"].append(out.values)
            
                pbar.update(len(inputs))
        end_retrieval_time = time.time()
        print("Retrieval time:", end_retrieval_time-start_retrieval_time, "QPS", len(questions_dataset)/(end_retrieval_time-start_retrieval_time))
        
        mem_t_s = time.time()
        indices_cpu_per_device = []
        values_cpu_per_device = []

        for d_id, out in results.items():
            indices_cpu_per_device.append(torch.stack(out["indices"]).cpu().tolist())
            if return_scores:
                values_cpu_per_device.append(torch.stack(out["values"]).cpu().tolist())
        
        # correct the q_id order
        indices_cpu = []
        values_cpu = []
        
        for j in range(len(indices_cpu_per_device[0])):
            for d_id in range(len(indices_cpu_per_device)):
                if j<len(indices_cpu_per_device[d_id]):
                    indices_cpu.append(indices_cpu_per_device[d_id][j])
                    if return_scores:
                        values_cpu.append(values_cpu_per_device[d_id][j])
                
        
        men_t_e = time.time()
        print("Mem transference time:", men_t_e-mem_t_s)
        
        #if return_scores:
        #    values_cpu = torch.concat(values_cpu, axis=0)
        # concat is slow
        return RetrievalOutput(ids=indices_cpu, scores=values_cpu, timmings=(len(questions_dataset)/(end_retrieval_time-start_retrieval_time), men_t_e-mem_t_s))

        
    def _single_retrieval(self, questions_list, question_func, collection, top_k, collect_at, profiling, return_scores):
        
        sparse_model= CSRSparseRetrievalModel(collection, top_k=top_k).to(self.devices[0])
        questions_dataset = QuestionDataset(questions_list, question_func, collection.shape[-1])
        
        dl = torch.utils.data.DataLoader(questions_dataset, 
                                            batch_size=len(self.devices), 
                                            pin_memory=True, 
                                            num_workers=0)
        
        start_retrieval_time = time.time()
        indices = []
        values = []
        for question in tqdm(dl):
            r = sparse_model(*[x.to(self.devices[0]) for x in question])
            indices.append(r.indices)
            if return_scores:
                values.append(r.values)
        end_retrieval_time = time.time()
        print("Retrieval time:", end_retrieval_time-start_retrieval_time, "QPS", len(questions_dataset)/(end_retrieval_time-start_retrieval_time))
        
        mem_t_s = time.time()
        indices_cpu = torch.stack(indices).cpu().tolist()
        values_cpu = None
        if return_scores:
            values_cpu = torch.stack(values).cpu().tolist()
        
        men_t_e = time.time()
        print("Mem transference time:", men_t_e-mem_t_s)
        
        return RetrievalOutput(ids=indices_cpu, scores=values_cpu, timmings=(len(questions_dataset)/(end_retrieval_time-start_retrieval_time), men_t_e-mem_t_s))
    

class CSRSparseRetrievalModel(torch.nn.Module):
    def __init__(self, sparse_collection, top_k = 10):
        super().__init__()
        #self.shape = sparse_collection.sparse_vecs, sparse_collection.shape
        self.crow = torch.nn.parameter.Parameter(sparse_collection.sparse_vecs[0], requires_grad=False)
        self.indice = torch.nn.parameter.Parameter(sparse_collection.sparse_vecs[1], requires_grad=False)
        self.values = torch.nn.parameter.Parameter(sparse_collection.sparse_vecs[2], requires_grad=False)
        self.collection_matrix = None#torch.sparse_csr_tensor(self.crow, self.indice, self.values, sparse_collection.shape)
        self.shape = sparse_collection.shape
        self.top_k = top_k
        
    def forward(self, indices, values):
        query = torch.sparse_coo_tensor(indices, values.squeeze(0), (self.shape[-1],), dtype=self.values.dtype).to_dense()

        #print(x.shape)
        collection_matrix = torch.sparse_csr_tensor(self.crow, self.indice, self.values, self.shape, dtype=self.values.dtype)
    
        return torch.topk(collection_matrix @ query, k=self.top_k, dim=0)
        #return x

class CSRSparseRetrievalDistributedModel(CSRSparseRetrievalModel):
        
    def forward(self, indices, values, size):
        
        query = torch.sparse_coo_tensor(indices[0, :size].unsqueeze(0), values[0,:size], (self.shape[-1],), dtype=self.values.dtype).to_dense()
        #print(x.shape)
        collection_matrix = torch.sparse_csr_tensor(self.crow, self.indice, self.values, self.shape, dtype=self.values.dtype)
    
        return torch.topk(collection_matrix @ query, k=self.top_k, dim=0)
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
        indices = torch.tensor(list(b.keys()))
        # TODO fix I need to use the same dtype of the collection
        values = torch.tensor(list(b.values()))
        
        return indices, values#{"indices": indices, "values": values}
    
    
class DistributedQuestionDataset(QuestionDataset):

    def __getitem__(self, idx):
        b = self.bow(self.questions[idx])
        indices = list(b.keys())
        values = list(b.values())
        
        return indices, values
    
def distributed_collate_fn(data):
    max_len = max([len(x[0]) for x in data])
    indices = []
    values = []
    sizes = []
    for x in data:
        sizes.append(len(x[0]))
        indices.append(x[0]+[0]*(max_len-len(x[0])))
        values.append(x[1]+[0]*(max_len-len(x[1])))
    return torch.tensor(indices), torch.tensor(values), torch.tensor(sizes)