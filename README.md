# Artifact "CompoST: A Benchmark for Analyzing the Ability of LLMs To Compositionally Interpret Questions in a QALD Setting"

## Setup

We suggest using Python 3.11 to run this artifact, older versions have not been tested. The following instructions should work for most mayor Linux distributions.

Start by setting up a virtual environment and installing the required packages. For this, run the following at the top level of a cloned version of this repository:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

If you want to execute a single Python file manually, you have to activate the virtual environment and add the `src/` subdirectory as `PYTHONPATH`:

```bash
source venv/bin/activate
export PYTHONPATH=./src/
python some/python/file.py
```

## Artifact Structure

* src/lemon/ - Source code for using the lexical entries. Taken from [https://github.com/ag-sc/neodudes](https://github.com/ag-sc/neodudes) and adapted for our use case.
    * resources/ - Used lexical entries (partially taken from [https://github.com/ag-sc/neodudes](https://github.com/ag-sc/neodudes), partially newly-added entries) and all CompoST dataset files
* src/dudes/ - Additional source code for SPARQL querying and to use the lexical entries. Taken from [https://github.com/ag-sc/neodudes](https://github.com/ag-sc/neodudes) and adapted for our use case.
* src/llm/compositionality/ - Source code for the CompoST dataset creation and the evaluation of the LLMs
  * dataset_generation.py - CompoST dataset generation
  * dataset.py - Classes and functions for loading the dataset
  * models/comp_model.py - LightningModule for training generative model on CompoST dataset
  * training.py - Training script for training model on CompoST dataset with DeepSpeed and PyTorch Lightning
  * hyperparameters.py - Hyperparameter optimization script for training model on CompoST dataset with Optuna, calls training.py
  * prompting_modules.py - Prompting modules for DSPy experiments
  * fewshot.py - Few-Shot experiments using DSPy
  * zeroshot.py - Zero-Shot experiments using DSPy
  * evaluation.py - Evaluation script for evaluating DSPy programs as well as fine-tuned models on CompoST dataset
* src/llm/utils.py - Utility functions, like setting up DSPy for the experiments
* requirements.txt - Python requirements of this project

## Replication Steps

### Dataset Generation

For generating the dataset, we used a filtered DBpedia snapshot from 2022-12 to only include properties that can be verbalized by our approach. The RDF data can be found in filtered_dbpedia_data_2022_12.tar.zst. 

1. Find basic patterns:
```bash
python src/llm/compositionality/dataset_generation.py --subgraph --depth 3 --breadth 3 --limit 10000 --endpoint "http://localhost:8890/sparql" --randrevprob 0.5 --forcelabels --threads 16 --entitylist src/lemon/resources/4_5_query_result.csv.zst --samples 1000
python src/llm/compositionality/dataset_generation.py --subgraph --depth 3 --breadth 2 --limit 10000 --endpoint "http://localhost:8890/sparql" --randrevprob 0.5 --forcelabels --threads 16 --entitylist src/lemon/resources/4_5_query_result.csv.zst --samples 1000
python src/llm/compositionality/dataset_generation.py --subgraph --depth 2 --breadth 3 --limit 10000 --endpoint "http://localhost:8890/sparql" --randrevprob 0.5 --forcelabels --threads 16 --entitylist src/lemon/resources/4_5_query_result.csv.zst --samples 1000
python src/llm/compositionality/dataset_generation.py --subgraph --depth 2 --breadth 2 --limit 10000 --endpoint "http://localhost:8890/sparql" --randrevprob 0.5 --forcelabels --threads 16 --entitylist src/lemon/resources/4_5_query_result.csv.zst --samples 1000
```

2. Create datasets from raw instance data:
```bash
python src/llm/compositionality/dataset_generation.py --subgraphs --split --samples 10 --paths src/lemon/resources/compositionality_samples_subgraph_*.jsonl.zst # replace paths with your newly-generated instance data
```

### Experiments
