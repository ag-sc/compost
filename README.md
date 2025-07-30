# Artifact "CompoST: A Benchmark for Analyzing the Ability of LLMs To Compositionally Interpret Questions in a QALD Setting" (ISWC 2025)

DOI: TBA

Preprint: [https://doi.org/10.48550/arXiv.2507.21257](https://doi.org/10.48550/arXiv.2507.21257)

Zenodo: [10.5281/zenodo.16312287](https://doi.org/10.5281/zenodo.16312287)

## Authors

- David Maria Schmidt [0000-0001-7728-2884](https://orcid.org/0000-0001-7728-2884)
- Raoul Schubert [0000-0001-7728-2884](https://orcid.org/0009-0009-7743-5401)
- Philipp Cimiano [0000-0002-4771-441X](https://orcid.org/0000-0002-4771-441X)

## Dataset

The dataset files are named "compositionality_subgraph_...", including the difficulty level, and can be found in [src/lemon/resources](https://github.com/ag-sc/compost/tree/main/src/lemon/resources). The *.jsonl files are in the OpenAI format and can be used, e.g., for fine-tuning. The *_fixed.json files contain all data of the respective difficulty level. The keys 'train', 'val', and 'test' contain the regular tasks while 'fixed_shots' contains the self-contained benchmark items (under 'train' and 'test') with examples providing enough information to deduce the answer to the respective question. 'powersetgraphs' contains information about how the various items correspond to each other using their IDs. For more information please refer to [src/llm/compositionality/dataset.py](https://github.com/ag-sc/compost/blob/main/src/llm/compositionality/dataset.py).

## Setup

We suggest using Python 3.11 to run this artifact, older versions have not been tested. The following instructions should work for most major Linux distributions.

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
* experiments/ - CSV files specifying the parameters for the experiments presented in the paper
* results/ - DSPy optimization results as well as evaluation results for the experiments presented in the paper
* filtered_dbpedia_data_2022_12.tar.zst - Filtered DBpedia snapshot from 2022-12 to only include properties that can be verbalized by our approach
* requirements.txt - Python requirements of this project

## Replication Steps

### Dataset Generation

For generating the dataset, we used a filtered DBpedia snapshot from 2022-12 to only include properties that can be verbalized by our approach. The RDF data can be found in filtered_dbpedia_data_2022_12.tar.zst. A corresponding local DBpedia endpoint is expected to be available via [http://localhost:8890/sparql](http://localhost:8890/sparql).

The "max" dataset in the code corresponds to the "hard" dataset in the paper.

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

A full local DBpedia endpoint is expected to be available via [http://localhost:8890/sparql](http://localhost:8890/sparql). The snapshot used for the paper can be found here: [https://databus.dbpedia.org/dbpedia/collections/dbpedia-snapshot-2022-12](https://databus.dbpedia.org/dbpedia/collections/dbpedia-snapshot-2022-12)

1. Start an Ollama server and download the models needed for the experiments:
```bash
ollama serve
ollama pull llama3.3
ollama pull phi4
ollama pull olmo2:7b-1124-instruct-q4_K_M
ollama pull qwen2.5-coder
```

2. Run the DSPy optimizations:
```bash
# Choose dataset for experiments
export TRAINDS=src/lemon/resources/compositionality_subgraph_easy_10_fixed.json
export TRAINDS=src/lemon/resources/compositionality_subgraph_medium_10_fixed.json
export TRAINDS=src/lemon/resources/compositionality_subgraph_max_10_fixed.json

#For OpenAI GPT experiments, remove the "--api http://127.0.0.1:11434"
python src/llm/compositionality/zeroshot.py --endpoint "http://localhost:8890/sparql" --subgraphs --rootpath ./ --trainpath $TRAINDS --api http://127.0.0.1:11434 --autoexpcsv $EXPERIMENTCSV # choose .csv from experiments folder
python src/llm/compositionality/fewshot.py --endpoint "http://localhost:8890/sparql" --subgraphs --rootpath ./ --trainpath $TRAINDS --api http://127.0.0.1:11434 --autoexpcsv $EXPERIMENTCSV # choose .csv from experiments folder
```

3. Run evaluation for all experiments:
```bash
python src/llm/compositionality/evaluation.py --endpoint "http://localhost:8890/sparql" --api http://127.0.0.1:11434 --evaltest --testpath $TRAINDS --subgraphs --autofind-rootpath ./ --autofind --autofind-inclexist --autofind-frac 1.0 --autofind-id 0
# For each evaluation csv file:
python src/llm/compositionality/evaluation.py --endpoint "http://localhost:8890/sparql" --reeval $EVALFILE --outpath $EVALFILE_evaluated.csv
```

4. Fine-tune OpenAI models using the .jsonl files in `src/lemon/resources`

5. Run evaluation for fine-tuned models:
```bash
export MODELNAME=ft:<model_name> # replace with your model name
python src/llm/compositionality/evaluation.py --endpoint "http://localhost:8890/sparql" --progpath ./$MODELNAME --model "$MODELNAME" --evaltest --subgraphs --testpath $TRAINDS
```

6. Fine-tune other models:
```bash
python src/llm/compositionality/hyperparameters.py --batchsize 1 --studyname "Compositionality OLMo" --optunafile "comp_optuna_olmo.log" --endpoint "http://localhost:8890/sparql" --instruct --subgraphs --model "allenai/OLMo-2-1124-7B-Instruct" --trainpath $TRAINDS --valpath $TRAINDS --testpath $TRAINDS
python src/llm/compositionality/hyperparameters.py --batchsize 1 --studyname "Compositionality Qwen" --optunafile "comp_optuna_qwen.log" --endpoint "http://localhost:8890/sparql" --instruct --subgraphs --model "Qwen/Qwen2.5-Coder-7B-Instruct"  --trainpath $TRAINDS --valpath $TRAINDS --testpath $TRAINDS
```

7. The fine-tuning process already returns evaluation scores (calculated as 1.0 - F1 score, i.e., a score of 0.8 corresponds to a F1 score of 0.2). However, to see the actual evaluation results per question, one can also evaluate fine-tuned models as follows:
```bash
export MODELPATH=./<model_name>.ckpt # replace with your model name
python src/llm/compositionality/evaluation.py --ftmodel $MODELPATH --trainpath $TRAINDS
```
