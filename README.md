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
Disclaimer, python3.11 fails to install some dependencies. The code was only tested on python 3.10.13
### Prepare the data

```bash
bash build_datasets.sh

```

### Run the benchmarks

```bash
bash benchmarks.sh

```