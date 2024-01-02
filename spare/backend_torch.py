from spare import TYPE
from spare.backend import AbstractBackend, RetrievalOutput
import torch
from safetensors.torch import save_file
from safetensors import safe_open
from tqdm import tqdm
import time
import numpy as np
import psutil
import gc
import concurrent.futures

def thread_inference_loop(sparse_model, top_k, device, question_dataset_class, data, bow, rank):
    
    sparse_model.top_k = top_k
    questions_dataset = question_dataset_class(data, bow, sparse_model.shape[-1])
    
    dl = torch.utils.data.DataLoader(questions_dataset, 
                                            batch_size=1, 
                                            pin_memory=True, 
                                            num_workers=0)
    
    start_retrieval_time = time.time()
    indices = []
    values = []
        
    for question in tqdm(dl, desc=f"Thread: {rank}"):
        r = sparse_model(*[x.to(device) for x in question])
        indices.append(r.indices)
        values.append(r.values)
    end_retrieval_time = time.time()
    #print("Retrieval time:", end_retrieval_time-start_retrieval_time, "QPS", len(questions_dataset)/(end_retrieval_time-start_retrieval_time))
    
    mem_t_s = time.time()
    indices_cpu = torch.stack(indices).cpu()#.tolist()
    values_cpu = None
    values_cpu = torch.stack(values).cpu()#.tolist()
    
    men_t_e = time.time()
    #print("Mem transference time:", men_t_e-mem_t_s)
    
    return rank,RetrievalOutput(ids=indices_cpu, scores=values_cpu, timmings=(len(questions_dataset)/(end_retrieval_time-start_retrieval_time), men_t_e-mem_t_s))


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
            TYPE.uint8: torch.uint8,
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
    
    def get_available_memory_per_device_inGB(self):
        if torch.cuda.is_available():
            return torch.cuda.mem_get_info(0)[0] * 1e-9
        else:
            return psutil.virtual_memory().available/(1024**3)
    
    def preprare_for_dp_retrieval(self, collection):
        return CSRSparseRetrievalDistributedModel(collection).to(self.devices[0]), DistributedQuestionDataset

    def preprare_for_dp_retrieval_iterative(self, collection, objective):
        
        csc_tensor = torch.sparse_csr_tensor(*collection.sparse_vecs, collection.shape).to_sparse_csc()
    
        ccol = csc_tensor.ccol_indices()
        rindices = csc_tensor.row_indices()
        cvalues = csc_tensor.values()
        
        models = {device: CSRSparseRetrievalModelIterativeThreadSafe(ccol, 
                                                                     rindices, 
                                                                     cvalues, 
                                                                     collection.shape, 
                                                                     objective=objective).to(device) for device in self.devices}
        
        return models, QuestionDataset
    
    def dp_retrieval_iterative(self, sparse_models, dataset_class, questions_list, question_func, top_k, collect_at, profiling, return_scores):
        
        num_devices = len(self.devices)
        
        chunk_size = len(questions_list) // num_devices
        chunk_size += len(questions_list) % num_devices > 0

        question_chunks = [questions_list[i:i + chunk_size] for i in range(0, len(questions_list), chunk_size)]
        
        ids = [None] * len(self.devices)
        scores = [None] * len(self.devices)
        timings = [None] * len(self.devices)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_devices) as executor:
            futures = [executor.submit(thread_inference_loop, sparse_models[device], top_k, device, dataset_class, data_chunk, question_func, _id)
                    for _id, (device, data_chunk) in enumerate(zip(self.devices, question_chunks))]

            for future in concurrent.futures.as_completed(futures):
                _id, out = future.result()
                ids[_id] = out.ids
                scores[_id] = out.scores
                timings[_id] = out.timmings 
            
        return RetrievalOutput(ids=torch.cat(ids, dim=0), 
                            scores=torch.cat(scores, dim=0),
                            timmings=max(timings))
        
        
        
    
    def dp_retrieval(self, sparse_model, dataset_class, questions_list, question_func, top_k, collect_at, profiling, return_scores):
        
        sparse_model.top_k = top_k
        questions_dataset = dataset_class(questions_list, question_func, sparse_model.shape[-1])
        
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
            indices_cpu_per_device.append(torch.stack(out["indices"]).cpu())
            if return_scores:
                values_cpu_per_device.append(torch.stack(out["values"]).cpu())
        
        # correct the q_id order
        indices_cpu = []
        values_cpu = []
        
        convert_s_time = time.time()
        for j in range(len(indices_cpu_per_device[0])):
            for d_id in range(len(indices_cpu_per_device)):
                if j<len(indices_cpu_per_device[d_id]):
                    indices_cpu.append(indices_cpu_per_device[d_id][j])
                    if return_scores:
                        values_cpu.append(values_cpu_per_device[d_id][j])
        convert_e_time = time.time() 
        
        men_t_e = time.time()
        print("Mem transference time:", men_t_e-mem_t_s, "Without tolist, convert time:", convert_e_time - convert_s_time)
        
        #if return_scores:
        #    values_cpu = torch.concat(values_cpu, axis=0)
        # concat is slow
        return RetrievalOutput(ids=indices_cpu, scores=np.array(values_cpu), timmings=(len(questions_dataset)/(end_retrieval_time-start_retrieval_time), men_t_e-mem_t_s))
    
    def preprare_for_forward_retrieval(self, collection):
        return CSRSparseRetrievalModel(collection).to(self.devices[0]), QuestionDataset
    
    def preprare_for_forward_retrieval_iterative(self, collection, objective):
        return CSRSparseRetrievalModelIterative(collection, objective=objective).to(self.devices[0]), QuestionDataset
        
    def forward_retrieval(self, sparse_model, dataset_class, questions_list, question_func, top_k, collect_at, profiling, return_scores):

        sparse_model.top_k = top_k
        questions_dataset = dataset_class(questions_list, question_func, sparse_model.shape[-1])
        
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
        indices_cpu = torch.stack(indices).cpu()#.tolist()
        values_cpu = None
        if return_scores:
            values_cpu = torch.stack(values).cpu()#.tolist()
        
        men_t_e = time.time()
        print("Mem transference time:", men_t_e-mem_t_s)
        
        return RetrievalOutput(ids=indices_cpu, scores=values_cpu, timmings=(len(questions_dataset)/(end_retrieval_time-start_retrieval_time), men_t_e-mem_t_s))
    
    def preprare_for_sharding_retrieval(self, collection, shards_count):
        return CSRSparseRetrievalModel(collection, splits_count=shards_count).to(self.devices[0]), QuestionDataset
    
    def preprare_for_sharding_retrieval_iterative(self, collection, shards_count, objective):
        return ShardedCSRSparseRetrievalModelIterative(collection, splits_count=shards_count, objective=objective).to(self.devices[0]), QuestionDataset
    
    def sharding_retrieval(self, sparse_model, dataset_class, questions_list, question_func, top_k, collect_at, profiling, return_scores):

        sparse_model.top_k = top_k
        questions_dataset = dataset_class(questions_list, question_func, sparse_model.shape[-1])
        
        dl = torch.utils.data.DataLoader(questions_dataset, 
                                            batch_size=len(self.devices), 
                                            pin_memory=True, 
                                            num_workers=0)
        
        start_retrieval_time = time.time()
        local_results = [[] for _ in range(len(dl))]

        for shard_index in range(sparse_model.num_shards):

            for q_index, (indices, values) in enumerate(tqdm(dl, desc=f"Retrieve shard {shard_index}/{sparse_model.num_shards-1}")):

                query = sparse_model.build_dense_query(indices, values)
                
                local_results[q_index].append(sparse_model.shard_forward(query, shard_index))

        for j in tqdm(range(len(local_results))):
            local_results[j] = sparse_model.merge_topk_results_torch(local_results[j])

        end_retrieval_time = time.time()
        print("Retrieval time:", end_retrieval_time-start_retrieval_time, "QPS", len(questions_dataset)/(end_retrieval_time-start_retrieval_time))
        
        mem_t_s = time.time()
        values, indices = list(zip(*local_results))

        indices_cpu = torch.stack(indices).cpu()
        values_cpu = torch.stack(values).cpu()
        
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


class CSRSparseRetrievalModelIterativeThreadSafe(torch.nn.Module):
    def __init__(self, 
                 ccol,
                 rindices,
                 cvalues,
                 shape, 
                 top_k=10, 
                 objective="accuracy"):
        
        super().__init__()
        
        if objective=="accuracy":
            self.storage_dtype = torch.float32
        elif objective=="half":
            self.storage_dtype = torch.float16
        elif objective=="performance":
            self.storage_dtype = torch.uint8
        else:
            raise RuntimeError(f"Objective mode {objective} is not supported in class CSRSparseRetrievalModelIterative")
        
        self.ccol = ccol
        self.rindices = rindices
        self.cvalues = cvalues
        
        self.top_k = top_k
        self.shape = shape
        
    def to(self, device):
        self._device = device
        self.ccol = self.ccol.to(device)
        self.rindices = self.rindices.to(device)
        self.cvalues = self.cvalues.to(device)
        return super().to(device)
    
    def forward(self, indices, values):
        
        indices = indices.squeeze(0)
        values = values.squeeze(0)
                
        accumulated = torch.zeros(self.shape[0], dtype=self.storage_dtype).to(indices.device)
        
        for i, (strat_idx, end_idx) in enumerate(zip(self.ccol[indices], self.ccol[indices+1])):
            v_indices = self.rindices[strat_idx: end_idx]
            
            v_values = (self.cvalues[strat_idx: end_idx] * values[i]).type(self.storage_dtype)

            accumulated.index_add_(0, v_indices, v_values)
            
        return torch.topk(accumulated, k=self.top_k, dim=0)



class CSRSparseRetrievalModelIterative(torch.nn.Module):
    def __init__(self, sparse_collection, top_k=10, objective="accuracy"):
        super().__init__()
        if objective=="accuracy":
            self.storage_dtype = torch.float32
        elif objective=="half":
            self.storage_dtype = torch.float16
        elif objective=="performance":
            self.storage_dtype = torch.uint8
        else:
            raise RuntimeError(f"Objective mode {objective} is not supported in class CSRSparseRetrievalModelIterative")

        #print("DEBUG: DATA TYPE", self.storage_dtype)
        
        print("Torch convert tensors from CSR to CSC")
        csc_tensor = torch.sparse_csr_tensor(*sparse_collection.sparse_vecs, sparse_collection.shape).to_sparse_csc()

        self.ccol = torch.nn.parameter.Parameter(csc_tensor.ccol_indices(), requires_grad=False)
        self.rindices = torch.nn.parameter.Parameter(csc_tensor.row_indices(), requires_grad=False)
        self.cvalues = torch.nn.parameter.Parameter(csc_tensor.values(), requires_grad=False)
        self.dummy_param = torch.nn.Parameter(torch.empty(0), requires_grad=True)
        self.top_k = top_k
        self.shape = sparse_collection.shape
        
    #def to(self, device):
    #    self._device = device
    #    self.ccol = self.ccol.to(device)
    #    self.rindices = self.rindices.to(device)
    #    self.cvalues = self.cvalues.to(device)
    #    return super().to(device)
    
    def forward(self, indices, values):
        
        indices = indices.squeeze(0)
        values = values.squeeze(0)
        
        #accumulated = torch.zeros(self.shape[0], dtype=self.storage_dtype).to(indices.device)
    
        #start_positions = self.ccol[indices]
        #end_positions = self.ccol[indices+1]

        #ranges = [torch.arange(s, e, device=indices.device, dtype=torch.int32) for s, e in zip(start_positions, end_positions)]
            
        #index_tensor = torch.cat(ranges)
        
        #v_indices = self.rindices.index_select(0, index_tensor)
        #v_values = self.cvalues.index_select(0, index_tensor).type(self.storage_dtype)

        #accumulated.index_add_(0, v_indices, v_values)

        #return torch.topk(accumulated, 10)
        
        accumulated = torch.zeros(self.shape[0], dtype=self.storage_dtype).to(indices.device)
        
        for i, (strat_idx, end_idx) in enumerate(zip(self.ccol[indices], self.ccol[indices+1])):
            v_indices = self.rindices[strat_idx: end_idx]
            
            # 0.5 is to help to round the values when converting from float32 to uint8
            v_values = (self.cvalues[strat_idx: end_idx] * values[i]).type(self.storage_dtype)

            accumulated.index_add_(0, v_indices, v_values)
            
        return torch.topk(accumulated, k=self.top_k, dim=0)
    
class CSRSparseRetrievalDistributedModelIterative(CSRSparseRetrievalModelIterative):
        
    def forward(self, indices, values, size):
        return super().forward(indices[0, :size].unsqueeze(0), values[0,:size].unsqueeze(0))


class ShardedCSRSparseRetrievalModel(torch.nn.Module):
    def __init__(self, sparse_collection, top_k=10, splits_count=1):
        super().__init__()
        
        crow, indices, values = sparse_collection.sparse_vecs
        # Split the CSR matrix into `splits_count` row-wise splits
        self.splits = self._split_csr(crow, indices, values, splits_count)
        self.num_shards = splits_count
        
        self.shape = sparse_collection.shape
        self.top_k = top_k
        
        # caching of the current shard in mem
        self._shard_current_index = -1
        self._shard_collection_matrix = None
        self._shard_start_row = -1
        
        del crow
        del indices
        del values
        del sparse_collection.sparse_vecs
        
    def to(self, device):
        self._device = device
        return super().to(device)
    
    def _split_csr(self, crow, indices, values, N):
        """
        Splits a CSR matrix represented by crow, indices, and values into N smaller matrices.
        
        Returns a list of tuples. Each tuple contains the crow, indices, and values arrays
        for one of the smaller matrices.
        """
        
        # Number of rows in the original matrix
        num_rows = crow.shape[0] - 1
        
        # Number of rows in each smaller matrix
        rows_per_split = num_rows // N
        
        # Handle case where num_rows is not exactly divisible by N
        # Distribute the remainder rows across the first few splits
        num_splits_with_extra_row = num_rows % N
        #print(num_splits_with_extra_row)
        
        splits = []
        
        start_row = 0
        for i in range(N):
            if i < num_splits_with_extra_row:
                end_row = start_row + rows_per_split + 1
            else:
                end_row = start_row + rows_per_split
                
            start_idx = crow[start_row].item()
            end_idx = crow[end_row].item()
            
            new_crow = crow[start_row:end_row+1] - start_idx
            new_indices = indices[start_idx:end_idx]
            new_values = values[start_idx:end_idx]
            
            splits.append((new_crow, new_indices, new_values, start_row, end_row))
            
            start_row = end_row
        
        return splits

    def merge_topk_results_torch(self, results):
        """
        Merges the top-k results from each split to get an overall top-k using PyTorch.

        Parameters:
        - results: A list of tuples. Each tuple contains two tensors: scores and indices.
                The list is of size `splits_count`.
        - k: The number of top elements to select.

        Returns:
        - A tuple containing two tensors: the top-k scores and their indices.
        """
        # Concatenate all scores and indices
        all_scores = torch.cat([res[0].squeeze() for res in results], dim=0)
        all_indices = torch.cat([res[1].squeeze() for res in results], dim=0)
        
        # Get overall top-k scores and indices
        top_scores, topk_indices = torch.topk(all_scores, self.top_k)
        top_indices = all_indices[topk_indices]

        return top_scores, top_indices
    
    def shard_forward(self, query, shard_index):
        if self._shard_current_index!=shard_index:
            
            del self._shard_collection_matrix
            
            crow, cindices, cvalues, start_row, end_row = self.splits[shard_index]
            crow = crow.to(self._device)
            cindices = cindices.to(self._device)
            cvalues = cvalues.to(self._device)
            
            collection_matrix = torch.sparse_csr_tensor(crow, cindices, cvalues, (end_row-start_row, self.shape[1]), dtype=cvalues.dtype)
            
            self._shard_current_index=shard_index
            self._shard_collection_matrix = collection_matrix
            self._shard_start_row = start_row
        
        dot_results = self._shard_collection_matrix @ query
        local_scores, local_indices = torch.topk(dot_results, k=self.top_k, dim=0)
        
        del dot_results
        #del collection_matrix
        #del crow
        #del cindices
        #del cvalues
        
        return (local_scores, local_indices+self._shard_start_row)
    
    def build_dense_query(self, indices, values):
        indices = indices.to(self._device)
        values = values.to(self._device)
        return torch.sparse_coo_tensor(indices, values.squeeze(0), (self.shape[-1],), dtype=self.splits[0][2].dtype).to_dense()#.to(self._device)
    
    def forward(self, indices, values):
        #print(len(self.splits))
        
        query = self.build_dense_query(indices, values)
        #shards_s_t = time.time()
        results = []
        for shard_index in range(len(self.splits)):
            results.append(self.shard_forward(query, shard_index))
        #shards_e_t = time.time()
        merged = self.merge_topk_results_torch(results, self.top_k)#results#torch.topk(results, k=self.top_k, dim=0)
        #merged_e_t = time.time()
        
        #print("Time to compute all shards",shards_e_t-shards_s_t, "time to merge",merged_e_t-shards_e_t)
        return merged

class ShardedCSRSparseRetrievalModelIterative(ShardedCSRSparseRetrievalModel):
    def __init__(self, sparse_collection, top_k=10, splits_count=1, objective="accuracy"):
        super().__init__(sparse_collection=sparse_collection, top_k=top_k, splits_count=splits_count)
        
        if objective=="accuracy":
            self.storage_dtype = torch.float32
        elif objective=="half":
            self.storage_dtype = torch.float16
        elif objective=="performance":
            self.storage_dtype = torch.uint8
        else:
            raise RuntimeError(f"Objective mode {objective} is not supported in class CSRSparseRetrievalModelIterative")

        print("Convert splits to CSR to CSC")
        for i in range(len(self.splits)):
            crow, cindices, cvalues, start_row, end_row = self.splits[i]
            self.splits[i] = None
            print("Torch convert tensors from CSR to CSC")
            csc_tensor = torch.sparse_csr_tensor(crow, cindices, cvalues, sparse_collection.shape).to_sparse_csc()

            csc_ccol = csc_tensor.ccol_indices()
            csc_rindices = csc_tensor.row_indices()
            csc_cvalues = csc_tensor.values()
            del csc_tensor
            del crow
            del cindices
            del cvalues
            
            self.splits[i] = (csc_ccol, csc_rindices, csc_cvalues, start_row, end_row)
            gc.collect()
    
    def build_dense_query(self, indices, values):
        indices = indices.squeeze(0).to(self._device)
        values = values.squeeze(0).to(self._device)
        
        return indices, values
     
    def shard_forward(self, query, shard_index):
        if self._shard_current_index!=shard_index:
            
            del self._shard_collection_matrix
            
            ccol, rindices, cvalues, start_row, end_row = self.splits[shard_index]
            ccol = ccol.to(self._device)
            rindices = rindices.to(self._device)
            cvalues = cvalues.to(self._device)
            
            collection_matrix = (ccol, rindices, cvalues)
            
            self._shard_current_index=shard_index
            self._shard_collection_matrix = collection_matrix
            self._shard_start_row = start_row
        else:
            ccol, rindices, cvalues = collection_matrix
        
        indices, values = query
        
        accumulated = torch.zeros(self.shape[0], dtype=self.storage_dtype).to(self._device)
        
        for i, (strat_idx, end_idx) in enumerate(zip(ccol[indices], ccol[indices+1])):
            v_indices = rindices[strat_idx: end_idx]
            
            # 0.5 is to help to round the values when converting from float32 to uint8
            v_values = (cvalues[strat_idx: end_idx] * values[i] + 0.5).type(self.storage_dtype)

            accumulated.index_add_(0, v_indices, v_values)
            
        local_scores, local_indices = torch.topk(accumulated, k=self.top_k, dim=0)
        
        del accumulated
        #del collection_matrix
        #del crow
        #del cindices
        #del cvalues
        
        return (local_scores, local_indices+self._shard_start_row)

class QuestionDataset(torch.utils.data.Dataset):
    def __init__(self, questions, bow, vocab_size):
        self.questions = questions#[:10000]
        self.bow = bow
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.questions)

    
    def __getitem__(self, idx):
        b = self.bow(self.questions[idx])
        #indices = torch.tensor(list(b.keys()))
        # TODO fix I need to use the same dtype of the collection
        #values = torch.tensor(list(b.values()))
        
        if len(b)>0:
            indices,  values = map(torch.tensor, zip(*b.items()))
        else:
            indices, values = torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32)

        return indices, values#{"indices": indices, "values": values}
    
    
class DistributedQuestionDataset(QuestionDataset):

    def __getitem__(self, idx):
        b = self.bow(self.questions[idx])
        
        if len(b)>0:
            indices,  values = map(list, zip(*b.items()))
        else:
            indices, values = [],[]
        #indices = list(b.keys())
        #values = list(b.values())
        return indices, values
    
def distributed_collate_fn(data):
    #print(data)
    max_len = max([len(x[0]) for x in data])
    indices = []
    values = []
    sizes = []
    for x in data:
        sizes.append(len(x[0]))
        indices.append(x[0]+[0]*(max_len-len(x[0])))
        values.append(x[1]+[0]*(max_len-len(x[1])))
    return torch.tensor(indices), torch.tensor(values), torch.tensor(sizes)