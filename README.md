# sparse-retrieval
python package for performing sparse-retrieval on accelerated hardware


## Reproduce benchmarks

### Installation


```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install .[benchmarks]
```

### Prepare the data

```bash
bash build_datasets.sh

```