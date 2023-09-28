from collections import defaultdict
from typing import Any

class BagOfWords:
    
    def __init__(self, tokenizer, dim) -> None:
        self.tokenizer = tokenizer
        self.dim = dim
    
    def __call__(self, text):
        bow = defaultdict(float)
        for t in self.tokenizer(text):
            bow[t]+=1.0
        return bow
