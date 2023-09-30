from backend import TYPE
import math


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