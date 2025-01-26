## Prerequisites

- A running MO instance or cluster with vector search and full text search enabled
- Python 3.8 or later

## Run the example

### Clone this repo

```bash
git clone git@github.com:ck89119/mo-vector-python.git
```

### Create a virtual environment

```bash
cd mo-vector-python/examples/orm-sqlalchemy-quickstart
python3 -m venv .venv
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Set the environment variables
Copy the `HOST`, `PORT`, `USERNAME`, `PASSWORD`, `DATABASE` to the `.env` file.

### Run this example

```text
$ python sqlalchemy-quickstart.py
Get 3-nearest neighbor documents:
  - distance: 0.00853986601633272
    document: fish
  - distance: 0.12712843905603044
    document: dog
  - distance: 0.7327387580875756
    document: tree
Get documents within a certain distance:
  - distance: 0.00853986601633272
    document: fish
  - distance: 0.12712843905603044
    document: dog
```