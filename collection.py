from typing import List
from backend import TYPE
from utils import get_coo_sparce_GB, get_csr_sparce_GB
from tqdm import tqdm
from metadata import MetaDataDFandDL
import os
import json
import shutil

class SparseCollection:
    ### implements a COO sparse matrix
    def __init__(self, 
                 collection_maxsize, 
                 text_to_vec=None,
                 vec_dim=None,
                 dtype=TYPE.float32, 
                 metadata=MetaDataDFandDL,
                 backend="torch") -> None:
        super().__init__()

        self.collection_maxsize = collection_maxsize
        if isinstance(dtype, int):  
            dtype = TYPE(dtype)
        self.dtype = dtype
        self.text_to_vec = text_to_vec
        
        if vec_dim is None and text_to_vec is None:
            raise RuntimeError(f"When creating a sparse collection from list of vectors you must provide vec_dim argument")
        elif vec_dim is None:
            self.vec_dim = text_to_vec.dim
        else:
            self.vec_dim = vec_dim
        
        if backend=="torch":
            from backend import TorchBackend
            self.backend = TorchBackend()
        else:
            RuntimeError("Only torch backend is currently supported")
        
        self.metadata = metadata()
            
    @classmethod
    def from_text_iterator(cls,
                           iterator,
                           collection_maxsize, 
                           text_to_vec, 
                           dtype=TYPE.float32, 
                           max_files_for_estimation=1000,
                           backend="torch",
                           **kwargs):
        
        assert text_to_vec is not None
        
        sparse_collection = cls(collection_maxsize, text_to_vec=text_to_vec, dtype=dtype, backend=backend, **kwargs)
        
        sparse_collection._build_sparse_collection(iterator, max_files_for_estimation=max_files_for_estimation)
        
        return sparse_collection
    
    @classmethod
    def from_vec_iterator(cls,
                           iterator,
                           collection_maxsize, 
                           vec_dim,
                           dtype=TYPE.float32, 
                           max_files_for_estimation=1000,
                           backend="torch",
                           **kwargs):
        
        assert vec_dim is not None
        
        sparse_collection = cls(collection_maxsize, vec_dim=vec_dim, dtype=dtype, backend=backend, **kwargs)
        
        sparse_collection._build_sparse_collection(iterator, max_files_for_estimation=max_files_for_estimation)
        
        return sparse_collection
    
    def transform(self, transform_operator):        
        self._correct_transform_operator(transform_operator).convert(*self.sparse_vecs, self.shape, self.nnz, self.metadata, self.dtype, self.backend)

    def _correct_transform_operator(self, transform_operator):
        return transform_operator.for_coo()
    
    def _build_sparse_collection(self, iterator, max_files_for_estimation):
        
        if self.text_to_vec is None:
            list_bow_for_estimation = [next(iterator) for _ in tqdm(range(max_files_for_estimation), desc="Size estimation")]
        else:
            list_bow_for_estimation = [self.text_to_vec(next(iterator)) for _ in tqdm(range(max_files_for_estimation), desc="Size estimation")]
        
        shape, density = self._get_matrix_estimations(list_bow_for_estimation)
        self.shape = shape
        self.density = density 
        
        mem_needed = self.get_sparce_matrix_space()
        
        elements_expected = int(shape[0] * shape[1] * density)
        print(f"We estimate that the collection matrix will have density of {density:.4f}, which requires {mem_needed} GB. Plus 0.5GB for overheads.")
        
        # make a verification if it fits the GPU mem and CPU! plus add strategies if doesnt
        print(f"Expected number of elements {elements_expected} for a shape {shape}")
        
        # lets store the vecs, since it easy to perform operation to the vecs.
        self.sparse_vecs = self._build_sparse_tensor(iterator, self.collection_maxsize, elements_expected, list_bow_for_estimation)
        del list_bow_for_estimation
        
    
    def _build_sparse_tensor(self, iterator, collection_maxsize, elements_expected, list_bow_for_estimation):
        
        sparse_vecs = self._init_sparse_vecs(elements_expected)
        
        element_index = 0

        index_docs = 0
        
        ## add values for the current processed documents in the list_bow_for_estimation list
        for doc_id, bow in list_bow_for_estimation:
            self.metadata.register_docid(index_docs ,doc_id)
            elements_added = self._update_sparse_vecs(*sparse_vecs, bow, element_index, index_docs)

            element_index += elements_added
            index_docs+=1

            del bow
        
        ## read the reminder of the collection
        for doc_id, bow_or_text in tqdm(iterator, total=collection_maxsize-len(list_bow_for_estimation), desc="Creating sparse matrix"):
            
            if index_docs>=collection_maxsize:
                break
            
            self.metadata.register_docid(index_docs ,doc_id)
            
            if self.text_to_vec is None:
                bow = bow_or_text
            else:
                bow = self.text_to_vec(bow_or_text)
            elements_added = self._update_sparse_vecs(*sparse_vecs, bow, element_index, index_docs)

            element_index += elements_added
            index_docs+=1
        
        self.nnz = element_index
        
        return self._slice_sparse_vecs(*sparse_vecs, element_index)
            
    def _slice_sparse_vecs(self, indices, values, element_index):
        # if I slice the indices I will create a non-continguous vector... which is not good
        #indices = self.backend.slice_tensor(indices, (slice(None, None, None), slice(None, element_index, None)))
        #values = self.backend.slice_tensor(values, slice(None, element_index, None))
        
        # for the COO we will pad the documents to +1
        doc_id = self.backend.get_value_from_tensor(indices, (0,element_index-1))+1
        len_pad = values.shape[0]-element_index
        self.backend.assign_data_to_tensor(indices,(0, slice(element_index, None, None)),[doc_id]*len_pad,TYPE.int64)
        
        # update shape to include the padded doc
        self.shape = (self.shape[0]+1, self.shape[1])
        self.metadata.update(doc_id, [], [0]) # pad the metadata
        
        return indices, values
    
    def _init_sparse_vecs(self, elements_expected):
        indices = self.backend.create_zero_tensor((2,elements_expected), TYPE.int64)
        values = self.backend.create_zero_tensor((elements_expected,), self.dtype)
        
        return indices, values
    
    def _sparcify_bow_and_meta_update(self, bow, index_docs):
        py_indices_row = []
        py_indices_col = []
        py_values = []
        for token_index in sorted(bow.keys()):
            py_indices_row.append(index_docs)
            py_indices_col.append(token_index)
            py_values.append(bow[token_index])
            
        self.metadata.update(index_docs, py_indices_col, py_values)
        
        return py_indices_row, py_indices_col, py_values
    
    def _update_sparse_vecs(self, indices, values, bow, element_index, index_docs):
        
        py_indices_row, py_indices_col, py_values = self._sparcify_bow_and_meta_update(bow, index_docs)
        
        self.backend.assign_data_to_tensor(indices, 
                                           (slice(None, None, None), slice(element_index,element_index+len(py_indices_col), None)),
                                           [py_indices_row, py_indices_col],
                                           TYPE.int64)
        self.backend.assign_data_to_tensor(values, 
                                           slice(element_index,element_index+len(py_indices_col), None),
                                           py_values,
                                           self.dtype)
            
        return len(py_indices_col)
    
    def get_sparce_matrix_space(self):
        return get_coo_sparce_GB(self.shape, self.density, self.dtype)
    
    def _get_matrix_estimations(self,
                               sampled_bow_list):
        

        dense_size = self.vec_dim*len(sampled_bow_list)
        density = sum([sum(bow.values()) for _, bow in sampled_bow_list])/dense_size
        shape = (self.collection_maxsize, self.vec_dim)

        return shape, density

    def _get_class_attributes(self):
        return {"nnz": self.nnz,
                      "shape": self.shape,
                      "collection_maxsize": self.collection_maxsize,
                      "dtype": self.dtype.value,
                      "vec_dim": self.vec_dim
                      }
    
    def save_to_file(self, folder_name):
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
        self.metadata.save_to_file(os.path.join(folder_name, "metadata.json"))
        self.backend.save_tensors_to_file(self.sparse_vecs, os.path.join(folder_name, "tensors.safetensors"))
        
        class_vars = self._get_class_attributes()
        
        with open(os.path.join(folder_name, "class_info.json"), "w") as f:
            json.dump(class_vars, f)
    
    @classmethod    
    def load_from_file(cls, folder_name, backend="torch"):
        
        with open(os.path.join(folder_name, "class_info.json")) as f:
            class_vars = json.load(f)
            
        nnz = class_vars.pop("nnz")
        shape = tuple(class_vars.pop("shape"))
        
        sparse_collection = cls(**class_vars, backend=backend)
        sparse_collection.nnz = nnz
        sparse_collection.shape = shape
        
        sparse_collection.metadata.load_from_file(os.path.join(folder_name, "metadata.json"))
        
        sparse_collection.sparse_vecs = sparse_collection.backend.load_tensors_from_file(os.path.join(folder_name, "tensors.safetensors"))
        
        return sparse_collection
        
class SparseCollectionCSR(SparseCollection):
    # sparse collection imlemented with csr
    def __init__(self, 
                 *args,
                 indices_dtype=TYPE.int32,
                 **kwargs,) -> None:
        super().__init__(*args, **kwargs)
        if isinstance(indices_dtype, int):  
            indices_dtype = TYPE(indices_dtype)
        self.indices_dtype = indices_dtype
        
    def _correct_transform_operator(self, transform_operator):
        return transform_operator.for_csr()
    
    def get_sparce_matrix_space(self):
        return get_csr_sparce_GB(self.shape, self.density, self.dtype, self.indices_dtype)
    
    def _init_sparse_vecs(self, elements_expected):
        crow_indices = self.backend.create_zero_tensor((self.shape[0]+1,), self.indices_dtype)
        col_indices = self.backend.create_zero_tensor((elements_expected,), self.indices_dtype)
        values = self.backend.create_zero_tensor((elements_expected,), self.dtype)
        
        return crow_indices, col_indices, values
    
    def _update_sparse_vecs(self, crow_indices, col_indices, values, bow, element_index, index_docs):
        
        py_indices_row, py_indices_col, py_values = self._sparcify_bow_and_meta_update(bow, index_docs)
        
        self.backend.assign_data_to_tensor(values, 
                                           slice(element_index,element_index+len(py_indices_row), None),
                                           py_values,
                                           self.dtype)
        
        self.backend.assign_data_to_tensor(col_indices, 
                                           slice(element_index,element_index+len(py_indices_row), None),
                                           py_indices_col,
                                           self.indices_dtype)
        
        self.backend.assign_data_to_tensor(crow_indices, 
                                           index_docs,
                                           element_index,
                                           self.indices_dtype)
            
        return len(py_indices_row)
    
    def _slice_sparse_vecs(self, crow_indices, col_indices, values, element_index):
        # CSR does not need pad!
        
        col_indices = self.backend.get_slice_from_tensor(col_indices,slice(None, element_index, None))
        values = self.backend.get_slice_from_tensor(values,slice(None, element_index, None))
        
        # add last elem to crow_indices
        self.backend.assign_data_to_tensor(crow_indices, crow_indices.shape[0]-1, self.nnz, self.indices_dtype)
        
        return crow_indices, col_indices, values
    
    def _get_class_attributes(self):
        return super()._get_class_attributes() | {"indices_dtype":self.indices_dtype.value}