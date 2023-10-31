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
    uint8 = 5

uint8 = TYPE.uint8
int32 = TYPE.int32
int64 = TYPE.int64
float16 = TYPE.float16
float32 = TYPE.float32

def type_to_str(datatype):
    if datatype==uint8:
        return "spare.uint8"
    elif datatype==int32:
        return "spare.int32"
    elif datatype==int64:
        return "spare.int64"
    elif datatype==float16:
        return "spare.float16"
    elif datatype==float32:
        return "spare.float32"
    else:
        raise RuntimeError(f"{datatype} is not valid")