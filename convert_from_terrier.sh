for f in beir_datasets/*
do
	echo "Processing $f"
    python indexing_from_terrier_mp.py $f --process 20
done