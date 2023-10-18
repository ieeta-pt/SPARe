for f in beir_datasets/*
do
	echo "Processing $f"

    python -m pyserini.index.lucene \
                    --collection JsonCollection \
                    --input $f/collection \
                    --index $f/anserini_index \
                    --generator DefaultLuceneDocumentGenerator \
                    --threads 1 \
                    --storeDocvectors \
                    --optimize

    python indexing_from_anserini.py $f
    #python converting_from_anserini.py $f
done