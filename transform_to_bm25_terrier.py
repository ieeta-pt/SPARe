
from collection import SparseCollectionCSR
from transformations import BM25Transform
import math

scr = SparseCollectionCSR.load_from_file("csr_msmarco")

LOG_E_OF_2 = math.log(2)
LOG_2_OF_E = 1 / LOG_E_OF_2

def log2(x):
    return math.log(x) * LOG_2_OF_E

#bug on idf fix it pls
def terrier_idf(x, collection_size):
    return log2((collection_size-x*0.5)/(x+0.5))

scr.transform(BM25Transform(k1=1.2, b=0.75, idf_weighting=terrier_idf))

scr.save_to_file(f"csr_msmarco_bm25_12_075_terrier")