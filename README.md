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

Note, you may also need to install python[your-version]-dev and java package if its not available on your system already. 
### Prepare the data

```bash
bash build_datasets.sh

```