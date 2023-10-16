from spare.backend import AbstractBackend, TYPE, RetrievalOutput
from safetensors.jax import save_file
from safetensors import safe_open
from tqdm import tqdm
import time
import jax
import jax.numpy as jnp

class JaxBackend(AbstractBackend):
    
    def __init__(self):
        super().__init__(jax.devices())
        
        self.types_converter = {
            TYPE.int32: jnp.int32,
            TYPE.int64: jnp.int64,
            TYPE.float32: jnp.float32,
            TYPE.float16: jnp.float16,
        }
        