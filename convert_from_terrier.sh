for f in beir_datasets/*
do
	echo "Processing $f"
    python build_terrier_index.py $f 
    python indexing_from_terrier_mp.py $f --process 20
done