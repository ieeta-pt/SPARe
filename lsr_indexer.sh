#!/bin/bash

lsr=deepct

for f in beir_datasets/*
do
    echo $f
    #python lsr_pisa_index.py $f $lsr
    python convert_lsr_bow_to_integer.py $f $lsr
    python -m pyserini.index.lucene   \
              --collection JsonVectorCollection   \
              --input $f/collection_$lsr\_integer  \
              --index $f/$lsr\_pyserini_index   \
              --generator DefaultLuceneDocumentGenerator   \
              --threads 12   \
              --impact \
              --pretokenized
    #python splade_pyterrier_index.py $f $lsr
    python spare_create_index_from_vec.py $f $lsr
done