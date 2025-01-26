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
cd mo-vector-python/examples/python-client-quickstart
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
$ python example.py
Downloading and loading the embedding model...
Search result ("a swimming animal"):
- text: "fish", distance: 0.45629183120475403
- text: "dog", distance: 0.6469335662074529
- text: "bb dog", distance: 0.7229517219644449
Search result ("a swimming animal"):
- text: "fish", distance: 0.45629183120475403
- text: "dog", distance: 0.6469335662074529
- text: "bb dog", distance: 0.7229517219644449
Search result ("a swimming animal, ['dog']"):
- text: "dog", score: 0.03252247488101534
- text: "bb dog", score: 0.031746031746031744
- text: "fish", score: 0.01639344262295082
Search result ("a swimming animal, ['dog']"):
- text: "fish", score: 0.2
- text: "dog", score: 0.1680668941203466
- text: "bb dog", score: 0.1024163823495667
```
