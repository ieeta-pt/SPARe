for f in beir_datasets/*
do
    for at in 10 100 1000 10000 100000
    do
        echo $f $at
        python benchmark_pyserini.py $f $at
        python benchmark_pyserini.py $f $at --threads 16
        python benchmark_pyterrier.py $f $at

        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyserini.py $f $at
        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at

        CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyserini.py $f $at
        CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at

        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyserini.py $f $at --fp_16
        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --fp_16

        CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyserini.py $f $at --fp_16
        CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --fp_16

        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyserini.py $f $at --cache_bow
        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --cache_bow

        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyserini.py $f $at --cache_bow --fp_16
        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --cache_bow --fp_16

        CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_pyserini.py $f $at
        CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_pyterrier.py $f $at
    done
done