from spare.collection import SparseCollection, SparseCollectionCSR
import spare
import json
from collections import defaultdict
import click

import glob
import re
import os

@click.command()
@click.argument("dataset_folder")
def main(dataset_folder):
    
    
    sparseCSR_collection = SparseCollectionCSR.from_bm25_pyserini_index(os.path.join(dataset_folder, "anserini_index"),
                                                                           k1=1.2,
                                                                           b=0.75,
                                                                           dtype=spare.float32,
                                                                           indices_dtype=spare.int32,
                                                                           backend="torch") 


    sparseCSR_collection.save_to_file(os.path.join(dataset_folder,f"csr_converted_anserini_bm25_12_075"))


if __name__ == '__main__':
    main()