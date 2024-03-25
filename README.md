# SPARe
⚠️ **Warning**: This repository is currently under development and it will be official presented on 25th March (ECIR 2024). Please note that certain features might be unstable, and the documentation may be incomplete during this phase. An alpha version with a stable version of the API is to be realeased soon.

## Overview
SPARe is an innovative approach to enhancing the efficiency of lexical retrievers through the utilization of GPU acceleration with sparse kernels. 

## Features
- **High-level API:**
  - **Sparse Collection API:** Manages interactions with backend compressed sparse row and column matrices.
  - **Searcher API:** Executes searches across the compressed sparse matrices using GPU.

- **Supported Backends:**
  - PyTorch
  - [Work in Progress] Jax

- **Compatibility:**
  - **Pyserini:** Converts Anserini indices into SPARe collections.
  - **PyTerrier:** Converts Terrier indices into SPARe collections. (Implementation is complete, but not included in the current version of the API.)

- **Additional Features to Note:**
  - Parallelism (SPMD, [Planned] Tensor)
  - [WIP] Hardware-aware searching: Optimizes retrieval strategies based on the available hardware resources.
  - Collection sharding: Allows collections to be divided into smaller segments to manage memory constraints more effectively (currently undergoing performance validation).


## Installation
Instructions on how to install and set up SPARe.

```bash
pip install git+https://github.com/ieeta-pt/SPARe.git
```

## Usage
Detailed guidelines on how to use SPARe in various scenarios.

### Sparse Collection API
#### Create sparse index directly from text
```python
import spare
from transformers import AutoTokenizer

# mock documents
docs = [{
        "id": "my first document",
        "contents": "This is my first document in my document collection"
    },{
        "id": "my second document",
        "contents": "This is another example of a shorter document"
    }]

# SPARe expects to always recieve a tuple with doc id and the document text.
collection_mapped = map(lambda doc: (doc["id"], doc["contents"]), docs)

# Simple tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Wraps the tokenizer to be compatible and with SPARe
bow = spare.BagOfWords(lambda x: tokenizer(x, add_special_tokens=False).input_ids, tokenizer.vocab_size)

# Creates a default counting sparse collection
collection = spare.SparseCollection.from_text_iterator(collection_mapped, 
                                                       text_to_vec=bow,
                                                       collection_maxsize=len(docs),
                                                       dtype=spare.float32,
                                                       backend="torch")

# save to a index folder
collection.save_to_file("my_two_document_collection")
```

#### Create sparse index from already created document vectors (usefull for LSR)
```python
import spare
from spare.metadata import MetaDataDocID

# same collection, but in the BOW format
[('my first document',
  {2023: 1.0,
   2003: 1.0,
   2026: 2.0,
   2034: 1.0,
   6254: 2.0,
   1999: 1.0,
   3074: 1.0}),
 ('my second document',
  {2023: 1.0,
   2003: 1.0,
   2178: 1.0,
   2742: 1.0,
   1997: 1.0,
   1037: 1.0,
   7820: 1.0,
   6254: 1.0})]

collection = spare.SparseCollection.from_vec_iterator(iter(bow_docs),
                                                       vec_dim=30522,
                                                       collection_maxsize=len(docs),
                                                       dtype=spare.float32,
                                                       backend="torch",
                                                       metadata=MetaDataDocID) # this defines the metadata that is stored, in this case its only the docID
                                                                               # by default SPARe uses MetaDataDFandDL, that stores docID, Doc Freq and Doc Length.
                                                                               # This is useful when loading weight values from LSR.

# save to a index folder
collection.save_to_file("my_two_document_collection")
```

#### Transformations
```python
import spare

# (...) Use any of the previous collections
collection = (...)

# Apply a static BM25 transformation to the collection. Note that for this transformation is required to have metadata of type MetaDataDFandDL.
collection.transform(spare.BM25Transform(k1=1.2, b=0.75))

# save to a index folder
collection.save_to_file("my_two_document_collection_converted_to_BM25")
```

### Searcher API

#### Search with cuSPARSE
```python
import spare

# lets load the previous bm25 collection to show how to load the collection
collection =  spare.SparseCollection.load_from_file("my_two_document_collection_converted_to_BM25")

# if our question are in text format we need to provide the *bow* function to the retrieval as well.
sparse_retriver = spare.SparseRetriever(collection, algorithm="dot")

question = {
  7820: 1.0,
  6254: 1.0,
}

sparse_retriver.retrieve([question], top_k=10, return_scores=True)
```

#### Search with iterative inner product algorithm
```python
import spare

# lets load the previous bm25 collection to show how to load the collection
collection =  spare.SparseCollection.load_from_file("my_two_document_collection_converted_to_BM25")

# if our question are in text format we need to provide the *bow* function to the retrieval as well.
sparse_retriver = spare.SparseRetriever(collection, algorithm="iterative")

question = {
  7820: 1.0,
  6254: 1.0,
}

sparse_retriver.retrieve([question], top_k=10, return_scores=True)
```

#### [THIS WILL BE CHANGED!!!] To change the execution datatype
```python
import spare

# lets load the previous bm25 collection to show how to load the collection
collection =  spare.SparseCollection.load_from_file("my_two_document_collection_converted_to_BM25")

# The execution datatype is controlled by the variable *objective* follwing the map:
#   accuracy: spare.float32
#   half: spare.float16
#   performance: spare.uint8
#
# This for sure will be change to dtype=spare.(...)
sparse_retriver = spare.SparseRetriever(collection, algorithm="iterative", objective="performance") # runs the accumulation with uint8

question = {
  7820: 1.0,
  6254: 1.0,
}

sparse_retriver.retrieve([question], top_k=10, return_scores=True)
```


## Contributing
Guidelines for contributing to SPARe will be made available soon. Please note that this project is currently maintained by a small team (me), and any assistance is greatly appreciated. Furthermore, I am open to collaborations.

## License
Currently, Apache License 2.0

## Citation
If you use SPARe in your research, please cite it using the following format:

```bibtex
@article{authors2024spare,
  title={SPARe: Supercharged Lexical Retrievers on GPU with Sparse Kernels},
  author={Almeida, Tiago
and Matos, S{\'e}rgio},
  journal={Journal/Conference},
  year={2024}
}
```

