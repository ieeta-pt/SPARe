import pyterrier as pt
if not pt.started():
    pt.init()
from pyterrier_pisa import PisaIndex

import pyt_splade
splade = pyt_splade.SpladeFactory()

dataset = pt.get_dataset('irds:msmarco-passage')
index = PisaIndex('./msmarco-passage-splade', stemmer='none')

# indexing
idx_pipeline = splade.indexing() >> index.toks_indexer()
idx_pipeline.index(dataset.get_corpus_iter(), batch_size=16)