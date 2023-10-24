from spare import TYPE
import math
import numpy as np



def maybe_init(class_or_insatnce):
    if isinstance(class_or_insatnce, type):
        return class_or_insatnce()
    else:
        return class_or_insatnce

def load_backend(backend):
    if backend=="torch":
        from spare.backend_torch import TorchBackend
        return TorchBackend()
    else:
        RuntimeError("Only torch backend is currently supported")

def get_best_np_dtype_for(min_value, max_value):
    uint32_bounds = np.iinfo("uint32")
    uint64_bounds = np.iinfo("uint64")
    int32_bounds = np.iinfo("int32")
    int64_bounds = np.iinfo("int64")
    
    if uint32_bounds.min <= min_value and max_value <= uint32_bounds.max:
        return np.uint32
    elif uint64_bounds.min <= min_value and max_value <= uint64_bounds.max:
        return np.uint64
    elif int32_bounds.min <= min_value and max_value <= int32_bounds.max:
        return np.int32
    elif int64_bounds.min <= min_value and max_value <= int64_bounds.max:
        return np.int64
    else:
        raise ValueError("Values are out of bounds for the specified dtypes.")

LOG_E_OF_2 = math.log(2)
LOG_2_OF_E = 1 / LOG_E_OF_2

def log2(x):
    return math.log(x) * LOG_2_OF_E


def terrier_idf(x, collection_size):
    # how terrier computes idf according to its java source code
    return log2((collection_size-x*0.5)/(x+0.5))

def idf_weighting(x, collection_size):
    #https://en.wikipedia.org/wiki/Okapi_BM25
    return math.log((collection_size-x+0.5)/(x+0.5)+1)

def get_bytes_in_dtype(dtype):
    if dtype == TYPE.float32:
        return 4
    elif dtype == TYPE.float16:
        return 2
    elif dtype == TYPE.int32:
        return 4
    elif dtype == TYPE.int64:
        return 8

def get_dense_sparce_GB(shape, dtype=TYPE.float32):
    bytes_per_value = get_bytes_in_dtype(dtype)
    
    return shape[0] * shape[1] * bytes_per_value *1e-9

def get_coo_sparce_GB(shape, density, dtype=TYPE.float32):
    bytes_per_value = get_bytes_in_dtype(dtype)
    
    cnt_sparce_elements = shape[0] * shape[1] * density
    
    return ((12+bytes_per_value)*cnt_sparce_elements )*1e-9
    
def get_csr_or_csc_sparce_GB(shape, density, dtype=TYPE.float32):
    bytes_per_value = get_bytes_in_dtype(dtype)
    
    cnt_sparce_elements = shape[0] * shape[1] * density
    
    c = min(shape[0], shape[1])
    
    return ((8+bytes_per_value)*cnt_sparce_elements + 8*c) *1e-9

def get_csr_sparce_GB(shape, density, dtype=TYPE.float32, indices_dtype=TYPE.int64):
    bytes_per_value = get_bytes_in_dtype(dtype)
    
    cnt_sparce_elements = shape[0] * shape[1] * density
    
    return ((get_bytes_in_dtype(indices_dtype)+bytes_per_value)*cnt_sparce_elements + get_bytes_in_dtype(indices_dtype)*shape[0]) *1e-9

def get_csc_sparce_GB(shape, density, dtype=TYPE.float32, indices_dtype=TYPE.int64):
    bytes_per_value = get_bytes_in_dtype(dtype)
    
    cnt_sparce_elements = shape[0] * shape[1] * density
    
    return ((get_bytes_in_dtype(indices_dtype)+bytes_per_value)*cnt_sparce_elements + get_bytes_in_dtype(indices_dtype)*shape[1]) *1e-9