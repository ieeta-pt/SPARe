"""
SPARe: SPARe: Supercharged lexical retrievers on GPU with sparse kernels	
"""
from enum import Enum

__version__="0.1.0"

class TYPE(Enum):
    int32 = 1
    int64 = 2
    float16 = 3
    float32 = 4
    
int32 = TYPE.int32
int64 = TYPE.int64
float16 = TYPE.float16
float32 = TYPE.float32