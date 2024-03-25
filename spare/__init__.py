"""
SPARe: SPARe: Supercharged lexical retrievers on GPU with sparse kernels	
"""

from spare.text2vec import BagOfWords
from spare.collection import SparseCollection
from spare.retriever import SparseRetriever
from spare.utils import TYPE
from spare.weighting_model import BM25Transform

__version__="0.1.1"

uint8 = TYPE.uint8
int32 = TYPE.int32
int64 = TYPE.int64
float16 = TYPE.float16
float32 = TYPE.float32

