: <<'END_COMMENT'
for f in beir_datasets/*
do
    for at in 10 100 1000
    do
        echo $f $at
        python benchmark_pisa.py $f $at
        python benchmark_pisa.py $f $at --threads 16

        #python benchmark_pyserini.py $f $at
        #python benchmark_pyserini.py $f $at --threads 16
        #python benchmark_pyterrier.py $f $at

        #CUDA_VISIBLE_DEVICES="0" python benchmark_sparse_retrieval_from_pyserini.py $f $at
        #CUDA_VISIBLE_DEVICES="0" python benchmark_sparse_retrieval_from_pyterrier.py $f $at

        #CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyserini.py $f $at
        #CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at

        #CUDA_VISIBLE_DEVICES="0" python benchmark_sparse_retrieval_from_pyserini.py $f $at --fp_16
        #CUDA_VISIBLE_DEVICES="0" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --fp_16

        #CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyserini.py $f $at --fp_16
        #CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --fp_16

        #CUDA_VISIBLE_DEVICES="0" python benchmark_sparse_retrieval_from_pyserini.py $f $at --cache_bow
        #CUDA_VISIBLE_DEVICES="0" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --cache_bow

        #CUDA_VISIBLE_DEVICES="0" python benchmark_sparse_retrieval_from_pyserini.py $f $at --cache_bow --fp_16
        #CUDA_VISIBLE_DEVICES="0" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --cache_bow --fp_16

        #CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_pyserini.py $f $at
        #CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --algorithm iterative --objective performance

        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyserini.py $f $at --algorithm iterative
        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --algorithm iterative

        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyserini.py $f $at --algorithm iterative --objective half
        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --algorithm iterative --objective half

        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyserini.py $f $at --algorithm iterative --objective performance
        CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --algorithm iterative --objective performance

        CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyserini.py $f $at --algorithm iterative
        CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --algorithm iterative

        CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyserini.py $f $at --algorithm iterative --objective half
        CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --algorithm iterative --objective half

        CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyserini.py $f $at --algorithm iterative --objective performance
        CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --algorithm iterative --objective performance

        CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_pyserini.py $f $at --algorithm iterative
        CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --algorithm iterative

        CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_pyserini.py $f $at --algorithm iterative --objective performance
        CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_pyterrier.py $f $at --algorithm iterative --objective performance

    done
done


for at in 10 100 1000
    do
    CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_splade.py $at

    CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_splade.py $at --algorithm iterative --objective half

    CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_splade.py $at --algorithm iterative --objective performance

    CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_splade.py $at 

    CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_splade.py $at --algorithm iterative --objective half

    CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_splade.py at --algorithm iterative --objective performance

    CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_splade.py $at 

    CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_splade.py $f $at --algorithm iterative

    CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_splade.py $f $at --algorithm iterative --objective performance
done

for f in beir_datasets/*
do
    for at in 10 100 1000 10000
    do
        echo $f $at
        python benchmark_pisa.py $f $at
        python benchmark_pisa.py $f $at --threads 16
    done
done
END_COMMENT

for f in beir_datasets/*
do
    for at in 10 100 1000 10000
    do
        echo $f $at
        python benchmark_pisa.py $f $at --threads 1
        python benchmark_pisa.py $f $at --threads 16
        python benchmark_pyterrier.py $f $at
        python benchmark_pyserini.py $f $at
        #python benchmark_pyserini_splade.py $at
        #python benchmark_pyserini_splade.py $at --threads 16
    done
done

at=1000
for f in beir_datasets/*
do
    echo $f $at
    python benchmark_pisa_splade.py $f $at --threads 1
    python benchmark_pisa_splade.py $f $at --threads 16

    python benchmark_pyserini_splade.py $f $at
    python benchmark_pyterrier_splade.py $f $at

    CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_splade.py $at

    CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_splade.py $at --algorithm iterative --objective half

    CUDA_VISIBLE_DEVICES="1" python benchmark_sparse_retrieval_from_splade.py $at --algorithm iterative --objective performance

    CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_splade.py $at 

    CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_splade.py $at --algorithm iterative --objective half

    CUDA_VISIBLE_DEVICES="0,1" python benchmark_sparse_retrieval_from_splade.py at --algorithm iterative --objective performance
done

for f in beir_datasets/*
do
    CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_splade.py $at 

    CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_splade.py $f $at --algorithm iterative

    CUDA_VISIBLE_DEVICES="" python benchmark_sparse_retrieval_from_splade.py $f $at --algorithm iterative --objective performance
done