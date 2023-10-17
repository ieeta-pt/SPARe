from spare import TYPE
from spare.backend import AbstractBackend, RetrievalOutput
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
            spare.int32: jnp.int32,
            spare.int64: jnp.int64,
            spare.float32: jnp.float32,
            spare.float16: jnp.float16,
        }
        