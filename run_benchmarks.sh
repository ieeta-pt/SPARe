for f in beir_datasets/*
do
    python benchmark_pyterrier.py $f
done