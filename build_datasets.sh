DATASET_PATH="beir_datasets"

python download_beir_dataset.py $DATASET_PATH

bash convert_from_anserini.sh

bash convert_from_terrier.sh

