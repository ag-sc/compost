import argparse
import csv
import glob
import io
import json
import logging
import math
import multiprocessing
import os
import random
import re
import statistics
import sys
import threading
import time
import traceback
from argparse import ArgumentParser
from collections import defaultdict
from csv import DictReader
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from pprint import pprint
from threading import Thread
from typing import Dict, Any, List, Tuple

import more_itertools
import networkx as nx
import numpy as np
import pandas as pd
import torch
import zstandard
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch
import seaborn as sns


# cache_path = Path(f"/dev/shm/{os.getpid()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
# cache_path.mkdir(parents=True, exist_ok=True)
#
# os.environ["DSPY_CACHEDIR"] = str(cache_path)
# os.environ["DSP_CACHEDIR"] = str(cache_path)
#
# import dspy  # type: ignore
# #from dspy import Evaluate
from more_itertools import chunked
from openai import OpenAI
from transformers import AutoTokenizer

from dudes import utils, consts
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from llm.compositionality.dataset import load_dataset, load_datasets, CompDataModule

from llm.utils import run_nvidia_smi

cache: Dict[str, Dict[str, Any]] = dict()
sparql_endpoint = SPARQLEndpoint(endpoint=consts.sparql_endpoint, cache=cache, check_endpoint=False, default_graph="http://dbpedia.org")

# while True:
#     try:
#         sparql_endpoint = SPARQLEndpoint(endpoint="https://dbpedia.org/sparql", cache=cache, check_endpoint=False)
#         break
#     except Exception as e:
#         print(e)
#         continue


class EvalScore(float):
    def __new__(cls, value, data=None):
        instance = super().__new__(cls, value)
        if data is None:
            data = dict()
        instance.data = data  # type: ignore
        return instance

# class EvalScore:
#     def __init__(self, score, data):
#         self.score = score
#         self.data = data
#
#     def __float__(self):
#         return self.score
#
#     def __int__(self):
#         return int(self.score)
#
#     def __mul__(self, other):
#         return self.score * other
#
#     def __sub__(self, other):
#         return self.score - other
#
#     def __add__(self, other):
#         return self.score + other
#
#     def __truediv__(self, other):
#         return self.score / other
#
#     def __floordiv__(self, other):
#         return self.score // other
#
#     def __mod__(self, other):
#         return self.score % other
#
#     def __pow__(self, other):
#         return self.score ** other
#

def eval_queries_raw(question, example, pred, data_id, data_subid, metadata=None):
    while True:
        try:
            estats, stats = utils.eval_queries(gold=example, pred=pred,
                                               sparql_endpoint=sparql_endpoint, debug=False)
            if metadata is None:
                metadata = dict()
            data = {
                "id": data_id,
                "subid": data_subid,
                "question": question,
                "sparql": example,
                "generated_sparql": pred,
                "TP": stats["True Positives"],
                "FP": stats["False Positives"],
                "FN": stats["False Negatives"],
                "Precision": stats["Precision"] if stats["Precision"] is not None else 0,
                "Recall": stats["Recall"] if stats["Recall"] is not None else 0,
                "F1": stats["F1"] if stats["F1"] is not None else 0
            } | metadata
            return EvalScore((stats["F1"] if stats["F1"] is not None else 0), data)
            # stats["F1"] if stats["F1"] is not None else 0
        except Exception as e:
            print(e)
            continue

def eval_queries(example, pred):
    while True:
        try:
            estats, stats = utils.eval_queries(gold=example.sparql_query, pred=pred.sparql_query,
                                               sparql_endpoint=sparql_endpoint, debug=False)
            data = {
                "id": example.id,
                "subid": example.subid,
                "question": example.question,
                "sparql": example.sparql_query,
                "generated_sparql": pred.sparql_query,
                "TP": stats["True Positives"],
                "FP": stats["False Positives"],
                "FN": stats["False Negatives"],
                "Precision": stats["Precision"] if stats["Precision"] is not None else 0,
                "Recall": stats["Recall"] if stats["Recall"] is not None else 0,
                "F1": stats["F1"] if stats["F1"] is not None else 0
            }
            return EvalScore((stats["F1"] if stats["F1"] is not None else 0), data)
            # stats["F1"] if stats["F1"] is not None else 0
        except Exception as e:
            print(e)
            continue

def eval_queries_openai(example, pred, data_id, data_subid, metadata=None):
    return eval_queries_raw(example[-2]["content"], example[-1]["content"], pred, data_id, data_subid, metadata=metadata)

def get_continuation_id(program_path, test_result_path=None):
    if test_result_path is None:
        test_result_path = str(Path(program_path).with_suffix('.csv'))
    start_id = -1
    start_subid = None
    if os.path.isfile(test_result_path):
        with open(test_result_path) as csv_file:
            for d in csv.DictReader(csv_file, delimiter=','):
                if int(d["id"]) > start_id:
                    start_id = int(d["id"])
                    if isinstance(d["subid"], str) and len(d["subid"]) == 0:
                        start_subid = None
                    else:
                        start_subid = d["subid"]
    return start_id, start_subid

def is_finished(program_path, arguments):
    try:
        test_result_path = arguments.outpath
        if test_result_path is None:
            test_result_path = str(Path(program_path).with_suffix('.csv'))

        test_path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "comp",
            "test-full.csv"
        )

        if arguments.evaltrain and arguments.trainpath is not None:
            test_path = arguments.trainpath
        elif arguments.evalval and arguments.valpath is not None:
            test_path = arguments.valpath
        elif arguments.evaltest and arguments.testpath is not None:
            test_path = arguments.testpath

        fixed_fewshots = False
        if "_fixed_" in os.path.basename(program_path):
            fixed_fewshots = True

        train, val, test = load_datasets(test_path=test_path, test_limit=arguments.limit, fixed_fewshot=fixed_fewshots, subgraphs=arguments.subgraphs)
        #test = load_dataset(test_path, limit=arguments.limit, fixed_fewshot=fixed_fewshots, subgraphs="test" if arguments.subgraphs else None)

        if arguments.subgraphs:
            if val == train:
                test = test + train
            else:
                test = test + val + train
            already_used = set()
            if os.path.isfile(test_result_path):
                with open(test_result_path) as csv_file:
                    already_used = {(int(d["id"]), int(d["subid"]), d["split"], d["question"]) for d in csv.DictReader(csv_file, delimiter=',')}

            print(f"Already used {already_used}", flush=True)

            test = [t for i, t in enumerate(test) if (int(t.id), int(t.subid), t.split, t.question) not in already_used]
            still_todo = {(int(t.id), int(t.subid), t.split, t.question) for t in test}

            print(f"Still todo {still_todo}", flush=True)
        else:
            start_id, start_subid = get_continuation_id(program_path, test_result_path)

            test = [t
                    for t in test
                    if t.id > start_id
                    and (
                            start_subid is None
                            or isinstance(start_subid, str)
                            or (
                                    t.subid is not None
                                    and start_subid is not None
                                    and not isinstance(t.subid, str)
                                    and t.subid > start_subid
                                )
                    )]

        return len(test) == 0
    except:
        return False



def eval_program(
        program_path,
        test_path=None,
        test_limit=None,
        test_result_path=None,
        num_threads=1,
        batch_size=30,
        api='http://localhost:11434',
        #fewshot=False,
        subgraphs=False,
):
    if "llama33" in os.path.basename(program_path):
        model = "ollama_chat/llama3.3"
    elif "llama32" in os.path.basename(program_path):
        model = "ollama_chat/llama3.2"
    elif "llama31" in os.path.basename(program_path):
        model = "ollama_chat/llama3.1"
    elif "phi4" in os.path.basename(program_path):
        model = "ollama_chat/phi4"
    elif "olmo2" in os.path.basename(program_path):
        model = "ollama_chat/olmo2:7b-1124-instruct-q4_K_M"
    elif "gpt-4o-mini-2024-07-18" in os.path.basename(program_path):
        model = "openai/gpt-4o-mini-2024-07-18"
    elif "gpt-4o-mini" in os.path.basename(program_path):
        model = "openai/gpt-4o-mini"
    elif "qwen" in os.path.basename(program_path):
        model = "ollama_chat/qwen2.5-coder"
    else:
        raise ValueError("Invalid model")

    sys.stdout = open(Path(program_path).with_suffix('.log'), "a+")
    sys.stderr = open(Path(program_path).with_suffix('.err'), "a+")

    cache_path = Path(os.path.join(
        os.path.dirname(program_path),
        os.path.basename(program_path)+"_evalcache" #re.sub(r"_[^_]+_[^_]+$", "", os.path.basename(program_path))+"_evalcache"
    ))
    cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["DSPY_CACHEDIR"] = str(cache_path)
    os.environ["DSP_CACHEDIR"] = str(cache_path)
    print(Path(program_path).with_suffix('.log'), flush=True)
    print(Path(program_path).with_suffix('.err'), flush=True)
    print(cache_path, flush=True)

    print("Evaluating", program_path, flush=True)

    # inputs = ["question"]
    # if "fixed" in os.path.basename(program_path):
    #     inputs = ["shot1_question", "shot1_sparql_query", "shot2_question", "shot2_sparql_query", "question"]

    #fewshot = False
    #if os.path.basename(program_path).startswith("fewshot"):
    #    fewshot = True
    fixed_fewshots = False
    if "_fixed_" in os.path.basename(program_path):
        fixed_fewshots = True
    import dspy
    if api is not None:
        lm = dspy.LM(model,
                     api_base=api,
                     api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
                     drop_params=True,
                     cache=False)
    else:
        lm = dspy.LM(model,
                     api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
                     drop_params=False if model.startswith("openai") else True,
                     cache=False)
    dspy.configure(lm=lm)
    # Load data
    if test_path is None:
        test_path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "comp",
            "test-full.csv"
        )

    train, val, test = load_datasets(test_path=test_path, test_limit=test_limit, fixed_fewshot=fixed_fewshots, subgraphs=subgraphs)
    #test = load_dataset(test_path, limit=test_limit, fixed_fewshot=fixed_fewshots, subgraphs=None)


    if test_result_path is None:
        test_result_path = str(Path(program_path).with_suffix('.csv'))
        # test_result_path = os.path.join(
        #     os.path.dirname(sys.modules["lemon"].__file__),
        #     "resources",
        #     f"compositionality_out_{os.path.basename(program_path)}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        # )

    try:
        prog = dspy.load(program_path)
    except Exception as e:
        print(e)
        return

    from dspy import Evaluate
    evaluate = Evaluate(
        devset=test,
        #metric=validate_context_and_answer,
        metric=no_eval,
        max_errors=2*len(test),#Ignore errors
        num_threads=num_threads,
        display_progress=True,
        display_table=True,
        provide_traceback=True,
        return_all_scores=True,
        return_outputs=True,
    )

    #overall_score, result_triples, individual_scores = evaluate(prog)  # [v.with_inputs("question") for v in val]

    # for example, prediction, score in result_triples:
    #     pass

    # start_id = -1
    # start_subid = None


    mode = "w"
    if os.path.isfile(test_result_path):
        mode = "a"
        # with open(test_result_path) as csv_file:
        #     for d in csv.DictReader(csv_file, delimiter=','):
        #         if int(d["id"]) > start_id:
        #             start_id = int(d["id"])
        #             if isinstance(d["subid"], str) and len(d["subid"]) == 0:
        #                 start_subid = None
        #             else:
        #                 start_subid = d["subid"]

    if subgraphs:
        if val == train:
            test = test + train
        else:
            test = test + val + train
        already_used = set()
        if os.path.isfile(test_result_path):
            with open(test_result_path) as csv_file:
                already_used = {(int(d["id"]), int(d["subid"]), d["split"], d["question"]) for d in csv.DictReader(csv_file, delimiter=',') if d["generated_sparql"] != "Error"}

        print(f"Already used {already_used}", flush=True)

        test = [t for i, t in enumerate(test) if (int(t.id), int(t.subid), t.split, t.question) not in already_used]
        still_todo = {(int(t.id), int(t.subid), t.split, t.question) for t in test}

        print(f"Still todo {still_todo}", flush=True)

        if len(test) == 0:
            print("No examples to evaluate left", flush=True)
            return

        with open(test_result_path, mode, newline='') as out_file:
            fieldnames = ['id', 'subid', 'split', 'question', 'sparql', 'generated_sparql', 'num_edges', 'num_edges_full', 'num_nodes',
                          'depth', 'breadth', 'base_depth', 'base_breadth', 'subgraphs', 'TP', 'FP', 'FN', 'Precision',
                          'Recall', 'F1']
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)

            if mode == "w":
                writer.writeheader()

            data_id = 0
            data_subid = None

            for chunk in chunked(test, batch_size):
                try:
                    overall_score, result_triples, individual_scores = evaluate(prog, devset=chunk)

                    for i, d in enumerate(individual_scores):
                        if isinstance(d, EvalScore):
                            example = chunk[i]
                            writer.writerow(d.data | {
                                "split": example.split,
                                "num_edges": example.num_edges,
                                "num_edges_full": example.num_edges_full,
                                "num_nodes": example.num_nodes,
                                "depth": example.depth,
                                "breadth": example.breadth,
                                "base_depth": example.base_depth,
                                "base_breadth": example.base_breadth,
                                "subgraphs": example.subgraphs,
                            })
                            out_file.flush()
                            pprint(d.data)
                        else:
                            example = chunk[i]
                            data = {
                                "id": example.id,
                                "subid": example.subid,
                                "question": example.question,
                                "sparql": example.sparql_query,
                                "generated_sparql": "Error",
                                "split": example.split,
                                "num_edges": example.num_edges,
                                "num_edges_full": example.num_edges_full,
                                "num_nodes": example.num_nodes,
                                "depth": example.depth,
                                "breadth": example.breadth,
                                "base_depth": example.base_depth,
                                "base_breadth": example.base_breadth,
                                "subgraphs": example.subgraphs,
                                "TP": None,
                                "FP": None,
                                "FN": None,
                                "Precision": 0,
                                "Recall": 0,
                                "F1": 0
                            }
                            writer.writerow(data)
                            out_file.flush()
                            pprint(data)
                # except Exception as e:
                #     print(e)
                #     print(traceback.format_exc())
                #     print("Chunk failed:", chunk)
                #     continue
                except:
                    print("Chunk failed:", chunk)
                    for example in chunk:
                        data = {
                            "id": example.id,
                            "subid": example.subid,
                            "question": example.question,
                            "sparql": example.sparql_query,
                            "generated_sparql": "Error",
                            "split": example.split,
                            "num_edges": example.num_edges,
                            "num_edges_full": example.num_edges_full,
                            "num_nodes": example.num_nodes,
                            "depth": example.depth,
                            "breadth": example.breadth,
                            "base_depth": example.base_depth,
                            "base_breadth": example.base_breadth,
                            "subgraphs": example.subgraphs,
                            "TP": None,
                            "FP": None,
                            "FN": None,
                            "Precision": 0,
                            "Recall": 0,
                            "F1": 0
                        }
                        writer.writerow(data)
                        out_file.flush()
                        pprint(data)
                    print(traceback.format_exc(), flush=True)
                    continue
    else:
        start_id, start_subid = get_continuation_id(program_path, test_result_path)

        print(f"Starting from {start_id} {start_subid}", flush=True)
        #test = [t for t in test if t.id > start_id and (start_subid is None or (t.subid is not None and start_subid is not None and t.subid > start_subid))]
        test = [t
                for t in test
                if t.id > start_id
                and (
                        start_subid is None
                        or isinstance(start_subid, str)
                        or (
                                t.subid is not None
                                and start_subid is not None
                                and not isinstance(t.subid, str)
                                and t.subid > start_subid
                        )
                )]

        if len(test) == 0:
            print("No examples to evaluate left", flush=True)
            return

        with open(test_result_path, mode, newline='') as out_file:

            fieldnames = ['id', 'subid', 'question', 'sparql', 'generated_sparql', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
            writer = csv.DictWriter(out_file, fieldnames=fieldnames)

            if mode == "w":
                writer.writeheader()

            for chunk in chunked(test, batch_size):
                try:
                    overall_score, result_triples, individual_scores = evaluate(prog, devset=chunk)

                    for i, d in enumerate(individual_scores):
                        if isinstance(d, EvalScore):
                            writer.writerow(d.data)
                            out_file.flush()
                            pprint(d.data)
                        else:
                            example = chunk[i]
                            data = {
                                "id": example.id,
                                "subid": example.subid,
                                "question": example.question,
                                "sparql": example.sparql_query,
                                "generated_sparql": "Error",
                                "TP": None,
                                "FP": None,
                                "FN": None,
                                "Precision": 0,
                                "Recall": 0,
                                "F1": 0
                            }
                            writer.writerow(data)
                            out_file.flush()
                            pprint(data)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print("Chunk failed:", chunk)
                    continue

    return test_result_path

def eval_openai_raw(
        basepath,
        model,
        test_path=None,
        test_limit=None,
        test_result_path=None,
        api=None
):

    sys.stdout = open(Path(basepath).with_suffix('.log'), "a+")
    sys.stderr = open(Path(basepath).with_suffix('.err'), "a+")

    cache_path = Path(os.path.join(
        os.path.dirname(basepath),
        os.path.basename(basepath)+"_evalcache" #re.sub(r"_[^_]+_[^_]+$", "", os.path.basename(program_path))+"_evalcache"
    ))
    cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["DSPY_CACHEDIR"] = str(cache_path)
    os.environ["DSP_CACHEDIR"] = str(cache_path)
    print(Path(basepath).with_suffix('.log'), flush=True)
    print(Path(basepath).with_suffix('.err'), flush=True)
    print(cache_path, flush=True)

    print("Evaluating", model, basepath, flush=True)

    import dspy
    if api is not None:
        lm = dspy.LM(model,
                     api_base=api,
                     api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
                     drop_params=True,
                     cache=True,
                     num_retries=20)
    else:
        lm = dspy.LM(model,
                     api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
                     drop_params=False if model.startswith("openai") else True,
                     cache=True,
                     num_retries=20)
    dspy.configure(lm=lm)
    # Load data
    if test_path is None:
        test_path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "comp",
            "openai_test_full.jsonl"
        )

    test = []
    with open(test_path, 'r') as testfile:
        for line in testfile:
            test.append(json.loads(line))

    if test_result_path is None:
        test_result_path = str(Path(basepath).with_suffix('.csv'))

    mode = "w"
    if os.path.isfile(test_result_path):
        mode = "a"


    start_id = -1
    if os.path.isfile(test_result_path):
        with open(test_result_path) as csv_file:
            start_id = len(list(csv.DictReader(csv_file, delimiter=',')))

    print(f"Starting from {start_id}", flush=True)
    test = [t for i, t in enumerate(test) if i >= start_id]

    if len(test) == 0:
        print("No examples to evaluate left", flush=True)
        return

    with open(test_result_path, mode, newline='') as out_file:
        fieldnames = ['id', 'subid', 'question', 'sparql', 'generated_sparql', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)

        if mode == "w":
            writer.writeheader()

        data_id = 0
        data_subid = 1
        if start_id > 0:
            data_id = int(start_id/3)
            data_subid = start_id%3+1

        for t in test:
            if test_limit is not None and data_id >= test_limit:
                break
            while True:
                try:
                    pred_queries = lm(messages=t["messages"][:-1], temperature=0.0)
                    pred_query = pred_queries[0]
                    d = eval_queries_openai(t["messages"], pred_query, data_id, data_subid)

                    if isinstance(d, EvalScore):
                        writer.writerow(d.data)
                        out_file.flush()
                        pprint(d.data)
                    else:
                        data = {
                            "id": data_id,
                            "subid": data_subid,
                            "question": t["messages"][1]["content"],
                            "sparql": t["messages"][2]["content"],
                            "generated_sparql": "Error",
                            "TP": None,
                            "FP": None,
                            "FN": None,
                            "Precision": 0,
                            "Recall": 0,
                            "F1": 0
                        }
                        writer.writerow(data)
                        out_file.flush()
                        pprint(data)
                    break
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print("Chunk failed:", t)
                    #no continue because we need to increment the ids

            if data_subid == 3:
                data_id += 1
                data_subid = 1
            else:
                data_subid += 1

    return test_result_path


def openai_subgraph_eval_thread(example):
    while True:
        try:
            if "shots" in example and len(example.shots) > 0:
                msgs = [
                    {"role": "system", "content": consts.openai_subgraph_prompt},
                ] + sum([
                    [
                        {"role": "user", "content": shot[0]},
                        {"role": "assistant", "content": shot[1]}
                    ] for shot in example.shots
                ], []) + [
                    {"role": "user", "content": example.question},
                    {"role": "assistant", "content": example.sparql_query}
                ]
            else:
                msgs = [
                    {"role": "system", "content": consts.openai_subgraph_prompt},
                    {"role": "user", "content": example.question},
                    {"role": "assistant", "content": example.sparql_query}
                ]
            print("Sending:", msgs[:-1], flush=True)
            print("Full:", msgs, flush=True)
            print("Example:", example, flush=True)
            pred_queries = lm(messages=msgs[:-1], temperature=0.0)
            pred_query = pred_queries[0]

            d = eval_queries_openai(msgs, pred_query, example.id, example.subid, metadata={
                "split": example.split,
                "num_edges": example.num_edges,
                "num_edges_full": example.num_edges_full,
                "num_nodes": example.num_nodes,
                "depth": example.depth,
                "breadth": example.breadth,
                "base_depth": example.base_depth,
                "base_breadth": example.base_breadth,
                "subgraphs": example.subgraphs,
            })

            if isinstance(d, EvalScore):
                return d.data
            else:
                data = {
                    "id": example.id,
                    "subid": example.subid,
                    "question": example.question,
                    "sparql": example.sparql_query,
                    "generated_sparql": "Error",
                    "split": example.split,
                    "num_edges": example.num_edges,
                    "num_edges_full": example.num_edges_full,
                    "num_nodes": example.num_nodes,
                    "depth": example.depth,
                    "breadth": example.breadth,
                    "base_depth": example.base_depth,
                    "base_breadth": example.base_breadth,
                    "subgraphs": example.subgraphs,
                    "TP": None,
                    "FP": None,
                    "FN": None,
                    "Precision": 0,
                    "Recall": 0,
                    "F1": 0
                }
                return data
        except Exception as e:
            print(e)
            print(traceback.format_exc())
            print("Chunk failed:", t)
            # no continue because we need to increment the ids

def initializer_openai_subgraph_eval_thread(basepath, model, api):
    sys.stdout = open(Path(basepath).with_suffix('.log'), "a+")
    sys.stderr = open(Path(basepath).with_suffix('.err'), "a+")

    cache_path = Path(os.path.join(
        os.path.dirname(basepath),
        os.path.basename(basepath) + "_evalcache" #f"{os.getpid()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_evalcache"
        # re.sub(r"_[^_]+_[^_]+$", "", os.path.basename(program_path))+"_evalcache"
    ))
    cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["DSPY_CACHEDIR"] = str(cache_path)
    os.environ["DSP_CACHEDIR"] = str(cache_path)
    print(Path(basepath).with_suffix('.log'), flush=True)
    print(Path(basepath).with_suffix('.err'), flush=True)
    print(cache_path, flush=True)

    import dspy
    global lm
    if api is not None:
        lm = dspy.LM(model,
                     api_base=api,
                     api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
                     drop_params=True,
                     cache=True,
                     num_retries=20)
    else:
        lm = dspy.LM(model,
                     api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
                     drop_params=False if model.startswith("openai") else True,
                     cache=True,
                     num_retries=20)
    dspy.configure(lm=lm)

    global client
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def eval_openai_raw_subgraph(
        basepath,
        model,
        test_path=None,
        test_limit=None,
        test_result_path=None,
        api=None,
        fixed_fewshots=False,
):

    sys.stdout = open(Path(basepath).with_suffix('.log'), "a+")
    sys.stderr = open(Path(basepath).with_suffix('.err'), "a+")

    cache_path = Path(os.path.join(
        os.path.dirname(basepath),
        os.path.basename(basepath)+"_evalcache" #re.sub(r"_[^_]+_[^_]+$", "", os.path.basename(program_path))+"_evalcache"
    ))
    cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["DSPY_CACHEDIR"] = str(cache_path)
    os.environ["DSP_CACHEDIR"] = str(cache_path)
    print(Path(basepath).with_suffix('.log'), flush=True)
    print(Path(basepath).with_suffix('.err'), flush=True)
    print(cache_path, flush=True)

    print("Evaluating", model, basepath, flush=True)

    # if api is not None:
    #     lm = dspy.LM(model,
    #                  api_base=api,
    #                  api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
    #                  drop_params=True,
    #                  cache=True,
    #                  num_retries=20)
    # else:
    #     lm = dspy.LM(model,
    #                  api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
    #                  drop_params=False if model.startswith("openai") else True,
    #                  cache=True,
    #                  num_retries=20)
    # dspy.configure(lm=lm)
    # Load data
    if test_path is None:
        test_path = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "compositionality_subgraph_hard_prompt_5.json.zst"
        )

    train, val, test = load_datasets(test_path=test_path, test_limit=test_limit, fixed_fewshot=fixed_fewshots, subgraphs=True)

    # evaldata = defaultdict(list)
    # with open(test_path, 'r') as testfile:
    #     evaldata = json.load(testfile)

    #if not fixed_fewshots or test_limit is not None:
    print("Train", len(train))
    print("Val", len(val))
    print("Test", len(test))

    if train == val:
        test = test + train
    else:
        test = test + val + train #[d | {"split": "test"} for d in evaldata["test"]] + [d | {"split": "val"} for d in evaldata["val"]] + [d | {"split": "train"} for d in evaldata["train"]]

    print("Comined", len(test), flush=True)

    #else:
    #    test = test + train

    if test_result_path is None:
        test_result_path = str(Path(basepath).with_suffix('.csv'))

    mode = "w"
    if os.path.isfile(test_result_path):
        mode = "a"

    start_id = -1
    already_used = set()
    if os.path.isfile(test_result_path):
        with open(test_result_path) as csv_file:
            already_used = {(int(d["id"]), int(d["subid"]), d["split"], d["question"]) for d in csv.DictReader(csv_file, delimiter=',')}

    print(f"Already used {already_used}", flush=True)

    test = [t for i, t in enumerate(test) if (int(t.id), int(t.subid), t.split, t.question) not in already_used]
    still_todo = {(int(t.id), int(t.subid), t.split, t.question) for t in test}

    print(f"Still todo {still_todo}", flush=True)

    if len(test) == 0:
        print("No examples to evaluate left", flush=True)
        return

    with open(test_result_path, mode, newline='') as out_file:
        fieldnames = ['id', 'subid', 'split', 'question', 'sparql', 'generated_sparql', 'num_edges', 'num_edges_full', 'num_nodes',
                      'depth', 'breadth', 'base_depth', 'base_breadth', 'subgraphs', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)

        if mode == "w":
            writer.writeheader()

        data_id = 0
        data_subid = None
        if not model.startswith("ft"):
            consts.openai_subgraph_prompt = consts.openai_subgraph_prompt_nonft

        with multiprocessing.Pool(processes=multiprocessing.cpu_count(), initializer=initializer_openai_subgraph_eval_thread, initargs=(basepath, model, api)) as pool:
            for row in pool.imap_unordered(openai_subgraph_eval_thread, test):
                writer.writerow(row)
                out_file.flush()

    return test_result_path

def _reevaluate_thr(t, res_dict):
    d = eval_queries_raw(t["question"], t["sparql"], t["generated_sparql"], t["id"], t["subid"])
    res_dict["result"] = d

def reevaluate(
        path,
        outpath,
):
    mode = "w"
    if os.path.isfile(outpath):
        mode = "a"

    already_used = set()
    if os.path.isfile(outpath):
        with open(outpath) as csv_file:
            already_used = {(int(d["id"]), int(d["subid"]), d["sparql"]) for d in csv.DictReader(csv_file, delimiter=',')}

    fieldnames = ['id', 'subid', 'question', 'sparql', 'generated_sparql', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1']
    with open(path, 'r') as testfile:
        dr = DictReader(testfile, delimiter=',')
        fieldnames = dr.fieldnames
        test = [d for d in dr if (int(d["id"]), int(d["subid"]), d["sparql"]) not in already_used]

    print(f"Already used {already_used}", flush=True)
    print(f"Still todo {len(test)}", flush=True)
    already_used.clear()

    manager = multiprocessing.Manager()

    with open(outpath, mode, newline='') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)

        if mode == "w":
            writer.writeheader()

        for t in test:
            while True:
                try:
                    print((t["question"], t["sparql"], t["generated_sparql"], t["id"], t["subid"]), flush=True)

                    return_dict = manager.dict()
                    return_dict["result"] = None
                    p = multiprocessing.Process(target=_reevaluate_thr, args=(t, return_dict))
                    p.start()
                    p.join(600)
                    if p.is_alive():
                        print("Process timed out, killing it")
                        p.kill()
                        p.join()
                    print("Process finished", flush=True)
                    d = return_dict.get("result")

                    #d = eval_queries_raw(t["question"], t["sparql"], t["generated_sparql"], t["id"], t["subid"])

                    if isinstance(d, EvalScore):
                        writer.writerow(t | d.data)
                        out_file.flush()
                        pprint(t | d.data)
                    else:
                        data = t | {
                            #"id": t["id"],
                            #"subid": t["subid"],
                            #"question": t["question"],
                            #"sparql": t["sparql"],
                            #"generated_sparql": t["generated_sparql"],
                            "TP": None,
                            "FP": None,
                            "FN": None,
                            "Precision": 0,
                            "Recall": 0,
                            "F1": 0
                        }
                        writer.writerow(data)
                        out_file.flush()
                        pprint(data)
                    break
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    print("Chunk failed:", t)
                    #no continue because we need to increment the ids



def fix_json(program_path, api='http://localhost:11434'):
    if "llama33" in os.path.basename(program_path):
        model = "ollama_chat/llama3.3"
    elif "llama32" in os.path.basename(program_path):
        model = "ollama_chat/llama3.2"
    elif "llama31" in os.path.basename(program_path):
        model = "ollama_chat/llama3.1"
    elif "phi4" in os.path.basename(program_path):
        model = "ollama_chat/phi4"
    elif "olmo2" in os.path.basename(program_path):
        model = "ollama_chat/olmo2:7b-1124-instruct-q4_K_M"
    elif "gpt-4o-mini-2024-07-18" in os.path.basename(program_path):
        model = "openai/gpt-4o-mini-2024-07-18"
    elif "gpt-4o-mini" in os.path.basename(program_path):
        model = "openai/gpt-4o-mini"
    elif "qwen" in os.path.basename(program_path):
        model = "ollama_chat/qwen2.5-coder"
    else:
        raise ValueError("Invalid model")

    jsonpath = str(Path(program_path).with_suffix('.json'))
    if os.path.isfile(jsonpath):
        print("Already fixed", jsonpath)
        return

    import dspy
    if api is not None and not model.startswith("openai"):
        lm = dspy.LM(model,
                     api_base=api,
                     api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
                     drop_params=True,
                     cache=True)
    else:
        lm = dspy.LM(model,
                     api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
                     drop_params=False if model.startswith("openai") else True,
                     cache=True)
    dspy.configure(lm=lm)

    try:
        prog = dspy.load(program_path)
        prog.save(jsonpath, save_program=False)
        print("Fixed", jsonpath)
    except Exception as e:
        print(e)
        return

def draw_comp_graph(G, res, title=None, filename=None):
    G = G.copy()
    # if res_map is not None:
    #     for e in G.edges:
    #         if G.edges[e]["label"] in res_map:
    #             G.edges[e]["label"] = res_map[G.edges[e]["label"]]
    #     for n in G.nodes:
    #         if "label" in G.nodes[n] and G.nodes[n]["label"] in res_map:
    #             G.nodes[n]["label"] = res_map[G.nodes[n]["label"]]

    for layer, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            G.nodes[node]["layer"] = -layer

    node_to_result = {int(r["subid"]): float(r["F1"]) for r in res}


    # Draw the graph
    #if topological:
    pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
    #else:
    #    pos = nx.bfs_layout(G, scale=1, start=sorted(G.nodes, key=lambda n: nx.descendants(G, n), reverse=True)[0])

    reduced_edges = [e for e in G.edges if abs(G.nodes[e[0]]["num_edges"] - G.nodes[e[1]]["num_edges"]) == 1]

    grouped_by_layer = defaultdict(list)
    for n in G.nodes:
        grouped_by_layer[G.nodes[n]["layer"]].append(n)

    max_x_coordinate = max([pos[n][0] for n in G.nodes])
    min_x_coordinate = min([pos[n][0] for n in G.nodes])

    num_edges_to_y = {
        int(G.nodes[n]['num_edges']): pos[n][1]
        for n in G.nodes
    }

    max_train_edges = max([int(r["num_edges"]) for r in res if r["split"] == "train"])
    min_train_edges = min([int(r["num_edges"]) for r in res if r["split"] == "train"])
    if len([int(r["num_edges"]) for r in res if r["split"] == "val"]) > 0:
        max_val_edges = max([int(r["num_edges"]) for r in res if r["split"] == "val"])
        min_val_edges = min([int(r["num_edges"]) for r in res if r["split"] == "val"])

        plt.vlines(x=min_x_coordinate - 0.05,
                   ymin=(num_edges_to_y[min_val_edges] + num_edges_to_y[min_val_edges - 1]) / 2,
                   ymax=(num_edges_to_y[max_val_edges] + num_edges_to_y[max_val_edges + 1]) / 2 if max_val_edges + 1 in num_edges_to_y else num_edges_to_y[max_val_edges] + abs(num_edges_to_y[min_val_edges] - num_edges_to_y[min_val_edges - 1]),
                   color='tab:grey',
                   linestyle='dotted')
    # else:
    #     max_val_edges = max_train_edges
    #     min_val_edges = min_train_edges
    if len([int(r["num_edges"]) for r in res if r["split"] == "test"]) > 0:
        max_test_edges = max([int(r["num_edges"]) for r in res if r["split"] == "test"])
        min_test_edges = min([int(r["num_edges"]) for r in res if r["split"] == "test"])

        plt.vlines(x=min_x_coordinate - 0.1,
                   ymin=(num_edges_to_y[min_test_edges] + num_edges_to_y[min_test_edges - 1]) / 2,
                   ymax=num_edges_to_y[max_test_edges] + abs(num_edges_to_y[min_test_edges] - num_edges_to_y[min_test_edges - 1]),
                   color='tab:grey',
                   linestyle='solid')

    plt.axhline(y=(num_edges_to_y[max_train_edges]+num_edges_to_y[max_train_edges+1])/2, color='tab:grey', linestyle='dashed')



    node_size = 1200/math.sqrt(len(G.nodes))
    #print("Node size:", node_size)

    all_red_nodes = [n for n in G.nodes if node_to_result[n] < 0.0001]
    bad_red = []#[n for n in all_red_nodes if all([node_to_result[d] > 0.99 for d in nx.descendants(G, n)])]
    bad_red_ext = [n for n in all_red_nodes if set(sum([[tuple(e) for e in G.nodes[d]["edgelist"]] for d in nx.descendants(G, n) if node_to_result[d] > 0.99], [])) == set(sum([[tuple(e) for e in G.nodes[n]["edgelist"]]], []))]
    bad_red_ext = list(set(bad_red_ext) - set(bad_red))
    good_red = list(set(all_red_nodes) - set(bad_red) - set(bad_red_ext))
    all_green_nodes = [n for n in G.nodes if node_to_result[n] > 0.99]
    bad_green = []#[n for n in all_red_nodes if all([node_to_result[d] < 0.0001 for d in nx.descendants(G, n)])]
    bad_green_ext = [n for n in all_green_nodes if any([node_to_result[d] < 0.0001 for d in nx.descendants(G, n)])]
    bad_green_ext = list(set(bad_green_ext) - set(bad_green))
    good_green = list(set(all_green_nodes) - set(bad_green) - set(bad_green_ext))

    orange_nodes = [n for n in G.nodes if node_to_result[n] >= 0.0001 and node_to_result[n] <= 0.99]


    # Draw nodes and edges
    nx.draw_networkx_nodes(G, pos, nodelist=bad_red_ext, node_color='firebrick', node_size=node_size)
    #nx.draw_networkx_nodes(G, pos, nodelist=bad_red, node_color='tab:pink', node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=good_red, node_color='red', node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=bad_green_ext, node_color='greenyellow', node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=good_green, node_color='lime', node_size=node_size)
    nx.draw_networkx_nodes(G, pos, nodelist=orange_nodes, node_color='orange', node_size=node_size)
    nx.draw_networkx_edges(G, pos, edgelist=reduced_edges, edge_color='black', width=1, alpha=0.5)

    for layer, nodes in grouped_by_layer.items():
        layer_y_pos = pos[nodes[0]][1]
        plt.text(max_x_coordinate+0.05, layer_y_pos, f"{G.nodes[nodes[0]]['num_edges']} Edge{'s' if G.nodes[nodes[0]]['num_edges'] > 1 else ''}", ha='left', va='center', size=20)#18*9/len(grouped_by_layer))

    legend_elements = [
        Patch(facecolor='lime', label='Correct Result'),
        Patch(facecolor='greenyellow', label='Correct Result, Comp Viol.'),
        Patch(facecolor='orange', label='Partially Correct Result'),
        Patch(facecolor='red', label='Wrong Result'),
        Patch(facecolor='firebrick', label='Wrong Result, Comp. Viol.'),
        # Line2D([0], [0], marker='o', color='tab:green', markerfacecolor='tab:green', markersize=15, label='Correct Result'),
        # Line2D([0], [0], marker='o', color='tab:red', markerfacecolor='tab:red', markersize=15, label='Wrong Result'),
        # Line2D([0], [0], marker='o', color='tab:red', markerfacecolor='tab:red', markersize=15, edgecolor='tab:pink', label='Wrong Result, Comp. Viol.'),
        # Line2D([0], [0], marker='o', color='tab:red', markerfacecolor='tab:red', markersize=15, edgecolor='tab:purple', label='Wrong Result, Comp. Viol. (Extended)'),
        Line2D([0], [0], color='tab:grey', linestyle='dashed', linewidth=1, label='Training Edge Count Limit'),
        Line2D([0], [0], color='tab:grey', linestyle='dotted', linewidth=1, label='Validation Edge Counts'),
        Line2D([0], [0], color='tab:grey', linestyle='solid', linewidth=1, label='Test Edge Counts'),
        #Patch(facecolor='orange', edgecolor='r', label='Color Patch')
    ]

    # Create the figure
    #fig, ax = plt.subplots()
    plt.legend(handles=legend_elements, loc='lower left', fontsize=16)
    if title is not None:
        plt.title(title, fontsize=20)

    plt.axis('off')
    plt.tight_layout()

    #plt.text(0.5, 0.5, "Hallo", ha='center', va='center', size=40)

    if filename is not None:
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(2560/100, 1440/100)
        plt.savefig(filename, dpi=300)
        plt.clf()
    else:
        try:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        except Exception as e:
            logging.error(f"Error: {str(e)}")
        # Show the plot
        plt.show()

@dataclass
class CompScores:
    red_nodes: int
    bad_red: int
    bad_red_ext: int
    good_red: int
    green_nodes: int
    good_green: int
    bad_green: int
    bad_green_ext: int
    orange_nodes: int
    all_nodes: int

    @property
    def fp(self):
        return self.green_nodes - self.good_green

    @property
    def tp(self):
        return self.good_green

    @property
    def fn(self):
        return self.bad_red_ext

    @property
    def tn(self):
        return self.good_red

    @property
    def prec(self):
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0

    @property
    def rec(self):
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0

    @property
    def fscore(self):
        return 2 * (self.prec * self.rec) / (self.prec + self.rec) if (self.prec + self.rec) > 0 else 0

    @property
    def comp_corrected_performance(self):
        return self.good_green / self.all_nodes

    @property
    def good_ratio(self):
        return (self.good_red+self.good_green)/self.all_nodes

    @property
    def bad_ratio(self):
        return (self.bad_red_ext+self.bad_green_ext)/self.all_nodes

def subgraph_comp_score(path, results_path, file_prefix="", file_postfix="", split=None, title=None):
    with open(path, 'r') as testfile:
        basedata = json.load(testfile)

    with open(results_path, 'r') as testfile:
        results = list(DictReader(testfile, delimiter=','))

    #id = task id (large pattern that is split up, i.e. one row in one of the base dataset files - id only unique for respective base dataset, i.e. (id, max_depth, max_breadth) is guaranteed to be unique only
    #gid/subid = identifier within the powerset graph, defines which other questions to check for compositionality score
    # base_/max_depth + breadth = useful for determining the powerset graph to use

    pgraphs = dict()
    pgraphs_data = dict()
    for gdata in basedata["powersetgraphs"]:
        pgraphs[(gdata["max_breadth"], gdata["max_depth"])] = nx.node_link_graph(json.loads(gdata["powersetgraph"]), edges="links")
        pgraphs_data[(gdata["max_breadth"], gdata["max_depth"])] = gdata
    #nx.descendants(pgraphs[(4,4)], [n for n in pgraphs[(4,4)].nodes if pgraphs[(4,4)].nodes[n]["gid"] == 95][0])

    grouped_graph_data = defaultdict(list)

    for r in results:
        grouped_graph_data[(int(r["id"]), int(r["base_breadth"]), int(r["base_depth"]))].append(r)

    drawdata = [(k,v) for k, v in grouped_graph_data.items()]
    drawdata.sort(key=lambda x: len(x[1]))

    red_nodes = []
    bad_red = []
    bad_red_ext = []
    good_red = []
    green_nodes = []
    good_green = []
    bad_green = []
    bad_green_ext = []
    orange_nodes = []
    all_nodes = []

    fixed_map = {
        "train": "val",
        "test": "test"
    }


    for k, mindata in drawdata:
        #print(k, len(mindata))
        G = pgraphs[k[1:]]
        res = mindata
        node_to_result = {int(r["subid"]): float(r["F1"]) for r in res}
        node_to_split = {int(r["subid"]): r["split"] if "fixed" not in results_path else fixed_map[r["split"]] for r in res}
        for n in G.nodes:
            if n not in node_to_result:
                node_to_result[n] = 1.0#for fixed, train samples are given in prompt and thus not evaluated separately
            if n not in node_to_split:
                node_to_split[n] = "train"

        all_nodes.extend(list(set([n for n in G.nodes if split is None or node_to_split[n] == split])))
        all_red_nodes = list(set([n for n in G.nodes if node_to_result[n] < 0.0001 and (split is None or node_to_split[n] == split)]))
        red_nodes.extend(all_red_nodes)
        new_bad_red = list(set([n for n in all_red_nodes if all([node_to_result[d] > 0.99 for d in nx.descendants(G, n)])]))
        bad_red.extend(new_bad_red)
        # bad_red_ext += list(set([n
        #                          for n in all_red_nodes
        #                          if set(sum([
        #         [tuple(e) for e in G.nodes[d]["edgelist"]]
        #         for d in nx.descendants(G, n)
        #         if node_to_result[d] > 0.99
        #     ], [])) == set(sum([[tuple(e) for e in G.nodes[n]["edgelist"]]], []))]))

        new_bad_red_ext = []
        for n in all_red_nodes:
            target_edges = set([tuple(e) for e in G.nodes[n]["edgelist"]])
            covered_edges = set()
            dsc = list(nx.descendants(G, n))
            for d in dsc:
                if node_to_result[d] > 0.99:
                    covered_edges.update([tuple(e) for e in G.nodes[d]["edgelist"]])
            if len(target_edges.difference(covered_edges)) == 0 or len(dsc) == 0:
                new_bad_red_ext.append(n)
        bad_red_ext.extend(new_bad_red_ext)

        new_good_red = list(set(all_red_nodes) - set(new_bad_red) - set(new_bad_red_ext))
        good_red.extend(new_good_red)

        all_green_nodes = list(set([n for n in G.nodes if node_to_result[n] > 0.99 and (split is None or node_to_split[n] == split)]))
        green_nodes.extend(all_green_nodes)

        new_bad_green = list(set([n for n in all_green_nodes if all([node_to_result[d] < 0.0001 for d in nx.descendants(G, n)])]))
        bad_green.extend(new_bad_green)

        new_bad_green_ext = list(set([n for n in all_green_nodes if any([node_to_result[d] < 0.0001 for d in nx.descendants(G, n)])]))
        bad_green_ext.extend(new_bad_green_ext)

        new_good_green = list(set(all_green_nodes) - set(new_bad_green) - set(new_bad_green_ext))
        good_green.extend(new_good_green)

        orange_nodes.extend(list(set([n for n in G.nodes if node_to_result[n] >= 0.0001 and node_to_result[n] <= 0.99 and (split is None or node_to_split[n] == split)])))

        if len(file_prefix+file_postfix) > 0:
            draw_comp_graph(pgraphs[k[1:]], mindata, filename=f"{file_prefix}{'_' if len(file_prefix) > 0 else ''}{k[0]}_{k[1]}_{k[2]}{'_' if len(file_postfix) > 0 else ''}{file_postfix}.pdf", title=title.title())

    #print(file_prefix)
    # print("Red nodes:", len(red_nodes))
    # print("Bad red nodes:", len(bad_red))
    # print("Bad red ext nodes:", len(bad_red_ext))
    # print("Good red nodes:", len(good_red))
    # print("Orange nodes:", len(orange_nodes))
    # print("Green nodes:", len(green_nodes))
    # print("Good green nodes:", len(good_green))
    # print("Bad green nodes:", len(bad_green))
    # print("Bad green ext nodes:", len(bad_green_ext))
    # print("All nodes:", len(all_nodes))

    scores = CompScores(
        red_nodes=len(red_nodes),
        bad_red=len(bad_red),
        bad_red_ext=len(bad_red_ext),
        good_red=len(good_red),
        green_nodes=len(green_nodes),
        good_green=len(good_green),
        bad_green=len(bad_green),
        bad_green_ext=len(bad_green_ext),
        orange_nodes=len(orange_nodes),
        all_nodes=len(all_nodes),
    )
    # print(scores)
    # print("Comp Corrected Performance:", scores.comp_corrected_performance)
    # print("Good Ratio:", scores.good_ratio)
    # print("Bad Ratio:", scores.bad_ratio)
    return scores



    # minlen = min([len(grouped_graph_data[k]) for k in grouped_graph_data])
    # minkey, mindata = [(k, grouped_graph_data[k]) for k in grouped_graph_data if len(grouped_graph_data[k]) == minlen][0]
    #draw_comp_graph(pgraphs[minkey[1:]], mindata)


def validate_context_and_answer(example, pred, trace=None):
    return eval_queries(example, pred)#["F1"]

def no_eval(example, pred, trace=None):
    data = {
        "id": example.id,
        "subid": example.subid,
        "question": example.question,
        "sparql": example.sparql_query,
        "generated_sparql": pred.sparql_query,
        "TP": 0,
        "FP": 0,
        "FN": 0,
        "Precision": 0,
        "Recall": 0,
        "F1": 0
    }
    return EvalScore(0, data)

def evaluate_finetuned_model(
        path,
        outpath=None,
        datalimit=None,
        datapath=None,
):
    from lightning import Trainer
    from lightning.pytorch.loggers import TensorBoardLogger
    from llm.compositionality.models.comp_model import CompModel
    instruct = True
    subgraphs = True
    if datapath is None:
        datapath = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources",
            "compositionality_subgraph_max_10_fixed.json"
        )
    if outpath is None:
        if path[-1] == "/":
            outpath = path[:-1] + ".csv"
        else:
            outpath = path + ".csv"

    if datalimit is None:
        datalimit = 0

    #sys.stdout = open(Path(path).with_suffix('.log'), "a+")
    #sys.stderr = open(Path(path).with_suffix('.err'), "a+")
    cache = dict()
    mat = re.search(r"comp_llm_(?P<model>[^_]+)_(?P<lr>[0-9.e-]+)_(?P<ld>[0-9.e-]+)_(?P<batchsize>[0-9]+)_(?P<epochs>[0-9]+)_(?P<date>.*).(ckpt|pt)", os.path.basename(path))
    assert mat is not None

    raw_model = str(mat.group("model"))
    if "olmo" in raw_model.lower():
        model_name = "allenai/OLMo-2-1124-7B" if not instruct else "allenai/OLMo-2-1124-7B-Instruct"
    elif "qwen" in raw_model.lower():
        model_name = "Qwen/Qwen2.5-Coder-7B" if not instruct else "Qwen/Qwen2.5-Coder-7B-Instruct"
    else:
        raise ValueError("Unknown model", raw_model)
    batch_size = int(mat.group("batchsize"))
    learning_rate = float(mat.group("lr"))
    ld = float(mat.group("ld"))
    epochs = int(mat.group("epochs"))
    print("Batch size: ", batch_size, flush=True)
    print("Learning rate: ", learning_rate, flush=True)
    print("Lambda: ", ld, flush=True)
    print("Epochs: ", epochs, flush=True)
    print("Model: ", model_name, flush=True)

    logger = TensorBoardLogger("/vol/neodudes/neodudes/tb_logs" if os.path.isdir("/vol/neodudes/neodudes") else "tb_logs",
                               name="comp_llm",
                               version=None if 'SLURM_ARRAY_TASK_ID' not in os.environ else os.environ['SLURM_ARRAY_TASK_ID'])

    trainer = Trainer(enable_checkpointing=False,
                      logger=logger,
                      #accelerator="cpu",
                      accelerator="cuda" if torch.cuda.is_available() else "cpu",
                      reload_dataloaders_every_n_epochs=1,
                      log_every_n_steps=1,
                      max_epochs=epochs,
                      min_epochs=epochs,
                      devices=-1,
                      strategy="deepspeed_stage_2_offload" if torch.cuda.is_available() else "ddp",
                      # strategy=DeepSpeedStrategy(
                      #     config=os.path.join(
                      #         os.path.dirname(sys.modules["lemon"].__file__),
                      #         "resources",
                      #         "deepspeed.json"
                      #     ),
                      # ),
                      precision="bf16-mixed",
                      # precision="bf16-mixed",
                      # precision="16-mixed",
                      # plugins=BitsandbytesPrecision(mode="int8"),
                      # plugins=BitsandbytesPrecision(mode="int8-training", dtype=torch.float16, ignore_modules={"lm_head"}),
                      # strategy="fsdp"
                      # strategy=FSDPStrategy(activation_checkpointing_policy={
                      #   torch.nn.TransformerEncoderLayer,
                      #   torch.nn.TransformerDecoderLayer,
                      # }, cpu_offload=True),
                      # accumulate_grad_batches=8,
                      )

    #with trainer.init_module():
    #model = CompModel.load_from_checkpoint(path, model_name=model_name,
    #                                       learning_rate=learning_rate, ld=ld, sparql_endpoint=None)#map_location=torch.device('cuda'), , strict=False

    #model.sparql_endpoint = SPARQLEndpoint(endpoint="http://dbpedia.org/sparql", cache=cache)

    tok = AutoTokenizer.from_pretrained(model_name)

    with open(datapath) as json_file:
        if datapath.endswith(".zst"):
            dctx = zstandard.ZstdDecompressor()
            stream_reader = dctx.stream_reader(json_file)
            json_file = io.TextIOWrapper(stream_reader, encoding='utf-8')
        data_json = json.load(json_file)
        data_raw = [
            de | {
                "input": tok.apply_chat_template([
                    {"role": "system", "content": consts.openai_subgraph_prompt},
                    {"role": "user", "content": de["question"]},
                ], tokenize=True, add_generation_prompt=True),
                "target": tok.apply_chat_template([
                    {"role": "system", "content": consts.openai_subgraph_prompt},
                    {"role": "user", "content": de["question"]},
                    {"role": "assistant", "content": de["query"]},
                ], tokenize=True, add_generation_prompt=False),
                "split": dataset,
                "subid": de["gid"]
            }
            for dataset in ["train", "val", "test"] for i, de in enumerate(data_json[dataset]) if datalimit == 0 or i < datalimit
        ]
        data_map = {
            (d["id"], d["subid"], tok.decode(d["input"], skip_special_tokens=True)): d
            for d in data_raw
        }
    print(f"Data loaded {len(data_map.keys())}", flush=True)

    dm = CompDataModule(
        tokenizer=tok,
        batch_size=64,
        datalimit=datalimit,
        is_instruct=True,
        is_subgraph=True,
        train_path=datapath,
        val_path=datapath,
        test_path=datapath
    )
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")


    model = CompModel(model_name=model_name,
                      learning_rate=learning_rate,
                      ld=ld,
                      sparql_endpoint=sparql_endpoint)#SPARQLEndpoint(endpoint="http://dbpedia.org/sparql", cache=cache))

    mode = "w"
    if os.path.isfile(outpath):
        mode = "a"

    fieldnames = ['id', 'subid', 'split', 'question', 'sparql', 'generated_sparql', 'num_edges', 'num_edges_full',
                  'num_nodes', 'depth', 'breadth', 'base_depth', 'base_breadth', 'subgraphs', 'TP', 'FP', 'FN',
                  'Precision', 'Recall', 'F1']

    rows = []

    ret_test = trainer.predict(model, dataloaders=dm.test_dataloader(), ckpt_path=path)  # , ckpt_path=path)

    for inputs, preds in ret_test:
        inputs_decoded = tok.batch_decode(inputs["input_token_ids"], skip_special_tokens=True)
        for inp, inpid, inpsubid, pred in zip(inputs_decoded, inputs["id"].tolist(), inputs["subid"].tolist(), preds):
            ref_data = data_map[(inpid, inpsubid, inp)]
            ref_data = ref_data | {
                "subid": ref_data["gid"],
                "split": "test",
                "sparql": ref_data["query"],
                "generated_sparql": pred[len(inp):],
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "Precision": 0,
                "Recall": 0,
                "F1": 0
            }

            row_data = {
                k: v for k, v in ref_data.items() if k in fieldnames
            }
            # writer.writerow(row_data)
            # out_file.flush()
            rows.append(row_data)
            print("Gold", ref_data["query"], flush=True)
            print("Pred", pred[len(inp):], flush=True)
    #print("Test", ret_test, flush=True)
    ret_val = trainer.predict(model, dataloaders=dm.val_dataloader(), ckpt_path=path)  # , ckpt_path=path)
    for inputs, preds in ret_val:
        inputs_decoded = tok.batch_decode(inputs["input_token_ids"], skip_special_tokens=True)
        for inp, inpid, inpsubid, pred in zip(inputs_decoded, inputs["id"].tolist(), inputs["subid"].tolist(), preds):
            ref_data = data_map[(inpid, inpsubid, inp)]
            ref_data = ref_data | {
                "subid": ref_data["gid"],
                "split": "val",
                "sparql": ref_data["query"],
                "generated_sparql": pred[len(inp):],
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "Precision": 0,
                "Recall": 0,
                "F1": 0
            }

            row_data = {
                k: v for k, v in ref_data.items() if k in fieldnames
            }
            # writer.writerow(row_data)
            # out_file.flush()
            rows.append(row_data)
            print("Gold", ref_data["query"], flush=True)
            print("Pred", pred[len(inp):], flush=True)

    # print("Val", ret_val)
    ret_train = trainer.predict(model, dataloaders=dm.train_dataloader(), ckpt_path=path)  # , ckpt_path=path)
    for inputs, preds in ret_train:
        inputs_decoded = tok.batch_decode(inputs["input_token_ids"], skip_special_tokens=True)
        for inp, inpid, inpsubid, pred in zip(inputs_decoded, inputs["id"].tolist(), inputs["subid"].tolist(), preds):
            ref_data = data_map[(inpid, inpsubid, inp)]
            ref_data = ref_data | {
                "subid": ref_data["gid"],
                "split": "train",
                "sparql": ref_data["query"],
                "generated_sparql": pred[len(inp):],
                "TP": 0,
                "FP": 0,
                "FN": 0,
                "Precision": 0,
                "Recall": 0,
                "F1": 0
            }

            row_data = {
                k: v for k, v in ref_data.items() if k in fieldnames
            }
            #writer.writerow(row_data)
            #out_file.flush()
            rows.append(row_data)
            print("Gold", ref_data["query"], flush=True)
            print("Pred", pred[len(inp):], flush=True)

    with open(outpath+f"_{os.getpid()}", mode, newline='') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        if mode == "w":
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

        # print("Train", ret_train)

def plot_heatmap(path, title=None, filename=None):
    grouped_by_edge_num = defaultdict(list)
    grouped_by_depth_breadth = defaultdict(list)
    max_breadth = 0
    min_breadth = 0
    max_depth = 0
    min_depth = 0

    path_scores = []
    with open(path) as f:
        data = list(csv.DictReader(f))

        test_scores = [float(d["F1"]) for d in data]
        if len(test_scores) == 0:
            return

        path_scores.append((statistics.mean(test_scores), path))

        # for d in test:
        #     grouped_by_edge_num[int(d['num_edges_full'])].append(float(d["F1"]))
        #     grouped_by_depth_breadth[(int(d['breadth']),int(d['depth']))].append(float(d["F1"]))
        #     max_breadth = max(max_breadth, int(d['breadth']))
        #     max_depth = max(max_depth, int(d['depth']))
        #     min_breadth = min(min_breadth, int(d['breadth']))
        #     min_depth = min(min_depth, int(d['depth']))

    path_scores.sort(key=lambda x: x[0], reverse=True)
    print("Path scores:")
    for score, path in path_scores:
        print(f"{score:.5f} {path}")

    with open(path_scores[0][1]) as f:
        data = list(csv.DictReader(f))

        test = [d for d in data]

        for d in test:
            grouped_by_edge_num[int(d['num_edges_full'])].append(float(d["F1"]))
            grouped_by_depth_breadth[(int(d['breadth']), int(d['depth']))].append(float(d["F1"]))
            max_breadth = max(max_breadth, int(d['breadth']))
            max_depth = max(max_depth, int(d['depth']))
            min_breadth = min(min_breadth, int(d['breadth']))
            min_depth = min(min_depth, int(d['depth']))

    bddata = [
        [
            []
            for _ in range(max_depth)
        ]
        for _ in range(max_breadth)
    ]

    for d in test:
        bddata[int(d['breadth']) - 1][int(d['depth']) - 1].append(float(d["F1"]))

    bddata_mean = [
        [
            statistics.mean(bddata[b][d]) if len(bddata[b][d]) > 0 else None
            for b in range(max_breadth)
        ]
        for d in range(max_depth)
    ]
    bddata_mean = bddata_mean

    for k, v in sorted(grouped_by_edge_num.items()):
        print(f"{k},{len(v)},{statistics.mean(v)}")

    df = pd.DataFrame(bddata_mean, columns=[i + 1 for i in range(max_breadth)], index=[d + 1 for d in range(max_depth)])
    df = df.fillna(value=np.nan)
    print(df.round(2))

    plt.figure(figsize=(10, 8))
    palette = sns.color_palette("coolwarm_r", as_cmap=True)
    heatmap = sns.heatmap(df, annot=True, fmt=".2f", cmap=palette, vmin=0.0, vmax=1.0, annot_kws={"size": 16}, cbar=False)
    #colorbar = heatmap.collections[0].colorbar
    #colorbar.ax.tick_params(labelsize=16)
    plt.xlabel('Breadth', fontsize=16)
    plt.ylabel('Depth', fontsize=16)
    if title is not None:
        plt.title(title, fontsize=16)
    # plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()

    if filename is not None:
        figure = plt.gcf()  # get current figure
        figure.set_size_inches(1000 / 100, 800 / 100)
        plt.savefig(filename, dpi=300)
        plt.clf()
    else:
        plt.show()

    for k, v in sorted(grouped_by_depth_breadth.items()):
        print(f"{k},{len(v)},{statistics.mean(v)}")

    print("Overall:", statistics.mean([float(d["F1"]) for d in test]))

def data_eval(data):
    macro_f1 = sum(float(d["F1"]) for d in data) / len(data)
    prec = sum(float(d["Precision"]) for d in data) / len(data)
    rec = sum(float(d["Recall"]) for d in data) / len(data)
    tp = sum(float(d["TP"]) if not isinstance(d["TP"], str) or len(d["TP"]) > 0 else 0 for d in data)
    fp = sum(float(d["FP"]) if not isinstance(d["FP"], str) or len(d["FP"]) > 0 else 0 for d in data)
    fn = sum(float(d["FN"]) if not isinstance(d["FN"], str) or len(d["FN"]) > 0 else 0 for d in data)
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    return macro_f1, prec, rec, tp, fp, fn, micro_f1

def gen_eval_summary(evalsummary, dataset, plot):
    dfdata = []

    grouped_by_approach_and_id: Dict[Tuple, Dict[int, List]] = defaultdict(lambda: defaultdict(list))

    comp_df_data = []

    by_prompting = defaultdict(list)

    for path in glob.glob(evalsummary + "*_evaluated.csv"):
        mipro_level = None
        if os.path.basename(path).startswith("ft:") and "fixed" in os.path.basename(path):
            shots = "fine-tuned (perfect shots)"
            optim = ""
            model = "gpt-4o-mini-2024-07-18"
            shots_fixed = "fixed"
            mode = "none"
        elif os.path.basename(path).startswith("ft:"):
            shots = "fine-tuned"
            optim = ""
            model = "gpt-4o-mini-2024-07-18"
            shots_fixed = "nonfix"
            mode = "none"
        else:
            parts = os.path.basename(path).split("_")
            shots = parts[0]
            optim = parts[1]
            if optim == "mipro":
                mipro_level = [p for p in parts if p in ["light", "medium", "heavy"]][0]
                optim = f"mipro-{mipro_level}"
            model = [p for p in parts if utils.any_in_list(["llama", "olmo", "phi", "gpt", "qwen"], p, lower=True)][0]
            shots_fixed = [p for p in parts if p in ["fixed", "nonfix"]][0] if shots == "fewshot" else "-"
            if shots_fixed == "fixed":
                shots = shots + " (perfect shots)"
            #print(shots, path)
            mode = "chainofthought" if "chainofthought" in parts else "none"

        if shots == "fewshot" and optim == "plain" and shots_fixed == "nonfix":
            continue

        # if shots_fixed == "fixed":
        #     continue

        #print(parts)

        approach_identifier = (shots, model, mode, optim, mipro_level, shots_fixed)

        #print(approach_identifier)  # shots, optim, model, shots_fixed)

        with open(path) as csv_file:
            csv_dict = csv.DictReader(csv_file, delimiter=',')
            data = [d for d in csv_dict if
                    not any([s.lower() in d["sparql"] for s in consts.compositionality_bad_props])]
            if shots_fixed == "nonfix" or shots == "zeroshot":
                for d in data:
                    grouped_by_approach_and_id[approach_identifier][int(d["id"])].append(d)

            # print(path, len(data))
            # macro_f1, prec, rec, tp, fp, fn, micro_f1
            full_eval = data_eval(data)
            train_eval = data_eval([d for d in data if d["split"] == "train"])
            if shots_fixed == "fixed":
                val_eval = data_eval([d for d in data if d["split"] == "train"])
            else:
                val_eval = data_eval([d for d in data if d["split"] == "val"])
            test_eval = data_eval([d for d in data if d["split"] == "test"])
            # print("Macro F1", macro_f1)
            # print("Micro F1", micro_f1)
            # print("Precision", prec)
            # print("Recall", rec)
            # add data to dataframe
            import dspy
            if not os.path.basename(path).startswith("ft:") and not os.path.isfile(os.path.join(os.path.dirname(path), "..", os.path.basename(path).replace(".csv_evaluated.csv", ".json"))):
                dp = dspy.load(os.path.join(os.path.dirname(path), "..", os.path.basename(path).replace(".csv_evaluated.csv", "")))
                dp.save(os.path.join(os.path.dirname(path), "..", os.path.basename(path).replace(".csv_evaluated.csv", ".json")), save_program=False)

            if not os.path.basename(path).startswith("ft:"):
                prompt = open(os.path.join(os.path.dirname(path), "..", os.path.basename(path).replace(".csv_evaluated.csv", ".json"))).read()
            else:
                prompt = consts.openai_subgraph_prompt

            if shots_fixed != "fixed":
                scores = subgraph_comp_score(dataset, path)
                train_scores = subgraph_comp_score(dataset, path, split="train")
                val_scores = subgraph_comp_score(dataset, path, split="val")
                test_scores = subgraph_comp_score(dataset, path, split="test")

                datadict = {
                    "Prompting": shots,
                    "Model": model,
                    "Mode": mode,
                    "Optimizer": optim,
                    "Macro F1": full_eval[0],
                    "Micro F1": full_eval[-1],
                    "Train Macro F1": train_eval[0],
                    "Train Micro F1": train_eval[-1],
                    "Val Macro F1": val_eval[0],
                    "Val Micro F1": val_eval[-1],
                    "Test Macro F1": test_eval[0],
                    "Test Micro F1": test_eval[-1],
                    "Comp F1": scores.fscore,
                    "Comp Corrected Performance": scores.comp_corrected_performance,
                    "Train Comp F1": train_scores.fscore,
                    "Train Comp Corrected Performance": train_scores.comp_corrected_performance,
                    "Val Comp F1": val_scores.fscore,
                    "Val Comp Corrected Performance": val_scores.comp_corrected_performance,
                    "Test Comp F1": test_scores.fscore,
                    "Test Comp Corrected Performance": test_scores.comp_corrected_performance,
                    "Resulting Prompt": prompt,
                    "path": path
                }
            else:
                scores = subgraph_comp_score(dataset, path)
                val_scores = subgraph_comp_score(dataset, path, split="val")
                test_scores = subgraph_comp_score(dataset, path, split="test")
                datadict = {
                    "Prompting": shots,
                    "Model": model,
                    "Mode": mode,
                    "Optimizer": optim,
                    "Macro F1": full_eval[0],
                    "Micro F1": full_eval[-1],
                    "Train Macro F1": train_eval[0],
                    "Train Micro F1": train_eval[-1],
                    "Val Macro F1": "-",
                    "Val Micro F1": "-",
                    "Test Macro F1": test_eval[0],
                    "Test Micro F1": test_eval[-1],
                    "Comp F1": scores.fscore,
                    "Comp Corrected Performance": scores.comp_corrected_performance,
                    "Train Comp F1": val_scores.fscore,
                    "Train Comp Corrected Performance": val_scores.comp_corrected_performance,
                    "Val Comp F1": "-",
                    "Val Comp Corrected Performance": "-",
                    "Test Comp F1": test_scores.fscore,
                    "Test Comp Corrected Performance": test_scores.comp_corrected_performance,
                    "Resulting Prompt": prompt,
                    "path": path
                }

            comp_df_data.append(datadict)
            by_prompting[shots].append(datadict)

    compdf = pd.DataFrame(comp_df_data, columns=[
        "Prompting",
        "Model",
        "Mode",
        "Optimizer",
        "Macro F1",
        "Micro F1",
        "Train Macro F1",
        "Train Micro F1",
        "Val Macro F1",
        "Val Micro F1",
        "Test Macro F1",
        "Test Micro F1",
        "Comp F1",
        "Comp Corrected Performance",
        "Train Comp F1",
        "Train Comp Corrected Performance",
        "Val Comp F1",
        "Val Comp Corrected Performance",
        "Test Comp F1",
        "Test Comp Corrected Performance",
        "Resulting Prompt",
    ])
    #print(compdf.round(2).to_string())
    compdf.round(2).sort_values(by=[
        "Prompting",
        "Macro F1",
        "Micro F1",
        "Comp F1",
        "Comp Corrected Performance",
    ],
    ascending=[
        True,
        False,
        False,
        False,
        False,
    ]).to_excel(os.path.join(evalsummary, "comp_eval_summary.xlsx"), index=False)

    best_dict = dict()
    for k, v in by_prompting.items():
        best = sorted(v, key=lambda x: x["Macro F1"], reverse=True)[0]
        print(best["path"])
        print("Best", best | {'Resulting Prompt': ""})
        if "max" in dataset:
            prefix = f"hard"
        elif "medium" in dataset:
            prefix = f"medium"
        elif "easy" in dataset:
            prefix = f"easy"
        else:
            print("Unknown dataset", dataset)
            prefix = None

        if best["Prompting"] == "zeroshot":
            prompting = "zero-shot"
        elif best["Prompting"] == "fewshot":
            prompting = "few-shot"
        elif best["Prompting"] == "fine-tuned":
            prompting = "fine-tuned"
        else:
            prompting = best["Prompting"]

        if "perfect" in best["Prompting"]:
            best_dict[k] = {
                "Prompting": best["Prompting"],
                # "Macro F1": f"${round(best['Macro F1'], 2)}$",
                # "Comp F1": f"${round(best['Comp F1'], 2)}$",
                # "Macro F1": f"${round(best['Macro F1'], 2)}$ (${round(best['Train Macro F1'], 2)}$, ${round(best['Val Macro F1'], 2)}$, ${round(best['Test Macro F1'], 2)}$)",
                # "Comp F1": f"${round(best['Comp F1'], 2)}$ (${round(best['Train Comp F1'], 2)}$, ${round(best['Val Comp F1'], 2)}$, ${round(best['Test Comp F1'], 2)}$)",
                # "Train Macro F1": "\multirow{2}{*}{"+f"${round(best['Train Macro F1'], 2)}$"+"}" if best["Train Macro F1"] != "-" else "",
                # "Train Comp F1": "\multirow{2}{*}{"+f"${round(best['Train Comp F1'], 2)}$"+"}" if best["Train Comp F1"] != "-" else "",
                # "Val Macro F1": "\multirow{2}{*}{"+f"${round(best['Val Macro F1'], 2)}$"+"}" if best["Val Macro F1"] != "-" else "",
                # "Val Comp F1": "\multirow{2}{*}{"+f"${round(best['Val Comp F1'], 2)}$"+"}" if best["Val Comp F1"] != "-" else "",
                # "Test Macro F1": "\multirow{2}{*}{"+f"${round(best['Test Macro F1'], 2)}$"+"}" if best["Test Macro F1"] != "-" else "",
                # "Test Comp F1": "\multirow{2}{*}{"+f"${round(best['Test Comp F1'], 2)}$"+"}" if best["Test Comp F1"] != "-" else "",
                "Train Macro F1": f"${round(best['Train Macro F1'], 2)}$" if best["Train Macro F1"] != "-" else "",
                "Train Comp F1": f"${round(best['Train Comp F1'], 2)}$" if best["Train Comp F1"] != "-" else "",
                "Val Macro F1": f"${round(best['Val Macro F1'], 2)}$" if best["Val Macro F1"] != "-" else "",
                "Val Comp F1": f"${round(best['Val Comp F1'], 2)}$" if best["Val Comp F1"] != "-" else "",
                "Test Macro F1": f"${round(best['Test Macro F1'], 2)}$" if best["Test Macro F1"] != "-" else "",
                "Test Comp F1": f"${round(best['Test Comp F1'], 2)}$" if best["Test Comp F1"] != "-" else "",
            }
        else:
            if plot:
                plot_heatmap(best["path"], title=f"{prompting.capitalize()} ({prefix.capitalize()})".title(), filename=f"{prefix}_{prompting}_heatmap.pdf")
                subgraph_comp_score(dataset, best["path"], file_prefix=prefix, file_postfix=prompting, title=f"{prompting.capitalize()} ({prefix.capitalize()})".title())
            best_dict[k] = {
                "Prompting": best["Prompting"],
                # "Macro F1": f"${round(best['Macro F1'], 2)}$",
                # "Comp F1": f"${round(best['Comp F1'], 2)}$",
                # "Macro F1": f"${round(best['Macro F1'], 2)}$ (${round(best['Train Macro F1'], 2)}$, ${round(best['Val Macro F1'], 2)}$, ${round(best['Test Macro F1'], 2)}$)",
                # "Comp F1": f"${round(best['Comp F1'], 2)}$ (${round(best['Train Comp F1'], 2)}$, ${round(best['Val Comp F1'], 2)}$, ${round(best['Test Comp F1'], 2)}$)",
                "Train Macro F1": f"${round(best['Train Macro F1'], 2)}$" if best["Train Macro F1"] != "-" else "",
                "Train Comp F1": f"${round(best['Train Comp F1'], 2)}$" if best["Train Comp F1"] != "-" else "",
                "Val Macro F1": f"${round(best['Val Macro F1'], 2)}$" if best["Val Macro F1"] != "-" else "",
                "Val Comp F1": f"${round(best['Val Comp F1'], 2)}$" if best["Val Comp F1"] != "-" else "",
                "Test Macro F1": f"${round(best['Test Macro F1'], 2)}$" if best["Test Macro F1"] != "-" else "",
                "Test Comp F1": f"${round(best['Test Comp F1'], 2)}$" if best["Test Comp F1"] != "-" else "",
            }
    return best_dict



if __name__ == "__main__":
    argparser = ArgumentParser()
    argparser.add_argument('--evalsummary', type=str, nargs="+")
    argparser.add_argument("--dataset", type=str, nargs="+")
    argparser.add_argument('--evalsummary-plots', action=argparse.BooleanOptionalAction)

    argparser.add_argument('--subgraphs', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--fixedshots', action=argparse.BooleanOptionalAction)

    argparser.add_argument('--autofind', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--dry-run', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--autofind-inclexist', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--autofind-fixjson', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--autofind-frac", type=float, default=1.0)
    argparser.add_argument("--autofind-id", type=int, default=0)
    argparser.add_argument("--autofind-model", type=str, nargs="+")
    argparser.add_argument("--autofind-exclmodel", type=str, nargs="+")
    argparser.add_argument("--autofind-rootpath", type=str, default=None)

    argparser.add_argument("--progpath", type=str, default="")
    argparser.add_argument("--model", type=str, default=None)
    argparser.add_argument("--outpath", type=str, default=None)
    argparser.add_argument('--evaltrain', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--evalval', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--evaltest', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--threads", type=int, default=8)
    argparser.add_argument("--limit", type=int, default=None)
    argparser.add_argument("--batchsize", type=int, default=30)
    argparser.add_argument("--trainpath", type=str, default=None)
    argparser.add_argument("--valpath", type=str, default=None)
    argparser.add_argument("--testpath", type=str, default=None)
    argparser.add_argument("--api", type=str, default=None)
    argparser.add_argument("--endpoint", type=str, default=None)
    argparser.add_argument("--reeval", type=str, default=None)
    argparser.add_argument("--ftmodel", type=str, default=None)

    #argparser.add_argument('--fewshot', action=argparse.BooleanOptionalAction)

    arguments = argparser.parse_args()

    if arguments.endpoint is not None:
        consts.endpoint = arguments.endpoint
        sparql_endpoint = SPARQLEndpoint(endpoint=arguments.endpoint, cache=dict(), check_endpoint=False, default_graph="http://dbpedia.org")

    assert arguments.autofind_frac > 0.0 and arguments.autofind_frac <= 1.0
    if arguments.reeval is not None:
        reevaluate(arguments.reeval, arguments.outpath)
        exit(0)
    elif arguments.ftmodel is not None:
        datapath = arguments.trainpath
        if arguments.valpath is not None:
            datapath = arguments.valpath
        if arguments.testpath is not None:
            datapath = arguments.testpath

        evaluate_finetuned_model(arguments.ftmodel, outpath=arguments.outpath, datalimit=arguments.limit, datapath=datapath)
        exit(0)
    elif arguments.evalsummary is not None and len(arguments.evalsummary) > 0:
        agg_res = defaultdict(lambda : defaultdict(dict))
        for es, ds in zip(arguments.evalsummary, arguments.dataset):
            if "easy" in ds:
                minds = "easy"
            elif "medium" in ds:
                minds = "medium"
            elif "max" in ds:
                minds = "hard"

            agg_res[minds] = gen_eval_summary(es, ds, plot=arguments.evalsummary_plots)

        df_rows = []
        for ds in ["easy", "medium", "hard"]:
            dsrow = [""] * 7
            dsrow[1] = ds
            df_rows.append(dsrow)
            for pr in ["zeroshot", "fewshot", "fine-tuned", "fewshot (perfect shots)", "fine-tuned (perfect shots)"]:
                row = [pr]
                for tvt in ["Train", "Val", "Test"]:
                    row.append(agg_res[ds][pr][f"{tvt} Macro F1"])
                for tvt in ["Train", "Val", "Test"]:
                    row.append(agg_res[ds][pr][f"{tvt} Comp F1"])
                df_rows.append(row)

        df = pd.DataFrame(df_rows, columns=["Prompting", "Train Macro F1", "Train Comp F1", "Val Macro F1", "Val Comp F1", "Test Macro F1", "Test Comp F1"])
        print(df.to_string(index=False))
        print(df.to_latex(index=False, escape=False))
        exit(0)

    paths = []
    if arguments.autofind:
        if arguments.autofind_rootpath is not None:
            basepath = arguments.autofind_rootpath
        else:
            basepath = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources"
            )

        zeroshots = [
            os.path.join(basepath, name)
            for name in os.listdir(basepath)
            if os.path.isdir(os.path.join(basepath, name))
               and name.startswith("zeroshot_")
               and not name.endswith("cache")
               and (not os.path.isfile(os.path.join(basepath, name+".csv")) or arguments.autofind_inclexist)
               and (arguments.autofind_model is None or len(arguments.autofind_model) == 0 or any([em in name for em in arguments.autofind_model]))
               and (arguments.autofind_exclmodel is None or len(arguments.autofind_exclmodel) == 0 or all([em not in name for em in arguments.autofind_exclmodel]))
               and (arguments.autofind_inclexist or not is_finished(os.path.join(basepath, name), arguments))
        ]
        fewshots = [
            os.path.join(basepath, name)
            for name in os.listdir(basepath)
            if os.path.isdir(os.path.join(basepath, name))
               and name.startswith("fewshot_") and not name.endswith("cache")
               and (not os.path.isfile(os.path.join(basepath, name+".csv")) or arguments.autofind_inclexist)
               and (arguments.autofind_model is None or len(arguments.autofind_model) == 0 or any([em in name for em in arguments.autofind_model]))
               and (arguments.autofind_exclmodel is None or len(arguments.autofind_exclmodel) == 0 or all([em not in name for em in arguments.autofind_exclmodel]))
               and (arguments.autofind_inclexist or not is_finished(os.path.join(basepath, name), arguments))
        ]
        paths = zeroshots + fewshots

        if arguments.autofind_fixjson:
            cache_path = Path("./dspy_fixcache")
            cache_path.mkdir(parents=True, exist_ok=True)

            os.environ["DSPY_CACHEDIR"] = str(cache_path)
            os.environ["DSP_CACHEDIR"] = str(cache_path)
            print(os.environ["DSPY_CACHEDIR"])
            for path in paths:
                fix_json(path, api=arguments.api)
            print("Fixed all jsons")
            exit(0)

        print("Found", paths)
        sample = list(more_itertools.chunked(paths, int(len(paths) * arguments.autofind_frac) + 1))[arguments.autofind_id]

        #sample = random.sample(paths, int(arguments.autofind_frac * len(paths)))
        print("Sampled", sample)
        #paths = sample
        paths = []

        for path in sample:
            parts = os.path.basename(path).split("_")
            shots = parts[0]
            optim = parts[1]
            mipro_level = "-"
            if optim == "mipro":
                mipro_level = [p for p in parts if p in ["light", "medium", "heavy"]][0]
            model = [p for p in parts if utils.any_in_list(["llama", "olmo", "phi", "gpt", "qwen"], p, lower=True)][0]
            shots_fixed = [p for p in parts if p in ["fixed", "nonfix"]][0] if shots == "fewshot" else "-"
            mode = "chainofthought" if "chainofthought" in parts else "none"

            if shots == "fewshot" and optim == "plain" and shots_fixed == "nonfix":
                continue

            # if shots_fixed == "fixed":
            #     continue

            paths.append(path)
    else:
        if arguments.progpath != "":
            if arguments.model is not None:
                paths = [(arguments.progpath, arguments.model)]
            else:
                paths = [arguments.progpath]
        else:
            print("No program path specified")
            argparser.print_help()

    print(paths)

    if arguments.dry_run:
        print("Dry run")
        exit(0)

    thread = threading.Thread(target=run_nvidia_smi, daemon=True)
    thread.start()

    threads = []

    for progpath in paths:
        print("Starting", progpath)
        if isinstance(progpath, Tuple):#eval_openai_raw_subgraph
            if arguments.subgraphs:
                if progpath is not None:
                    t = Process(target=eval_openai_raw_subgraph, kwargs={
                        "basepath": progpath[0],
                        "model": progpath[1],
                        "test_path": arguments.testpath,
                        "test_limit": arguments.limit,
                        "test_result_path": arguments.outpath,
                        "api": arguments.api,
                        "fixed_fewshots": arguments.fixedshots,
                    })
                    t.start()
                    threads.append(t)
                else:
                    print("Invalid Arguments")
                    argparser.print_help()
            else:
                if arguments.evaltrain and progpath is not None:
                    t = Process(target=eval_openai_raw, kwargs={
                        "basepath": progpath[0],
                        "model": progpath[1],
                        "test_path": arguments.trainpath,
                        "test_limit": arguments.limit,
                        "test_result_path": arguments.outpath,
                        "api": arguments.api,
                    })
                    t.start()
                    threads.append(t)
                elif arguments.evalval and progpath is not None:
                    t = Process(target=eval_openai_raw, kwargs={
                        "basepath": progpath[0],
                        "model": progpath[1],
                        "test_path": arguments.valpath,
                        "test_limit": arguments.limit,
                        "test_result_path": arguments.outpath,
                        "api": arguments.api,
                    })
                    t.start()
                    threads.append(t)
                elif arguments.evaltest and progpath is not None:
                    t = Process(target=eval_openai_raw, kwargs={
                        "basepath": progpath[0],
                        "model": progpath[1],
                        "test_path": arguments.testpath,
                        "test_limit": arguments.limit,
                        "test_result_path": arguments.outpath,
                        "api": arguments.api,
                    })
                    t.start()
                    threads.append(t)
                else:
                    print("Invalid Arguments")
                    argparser.print_help()
        else:
            if arguments.evaltrain and progpath is not None:
                t = Process(target=eval_program, kwargs={
                    "program_path": progpath,
                    "test_path": arguments.trainpath,
                    "test_limit": arguments.limit,
                    "test_result_path": arguments.outpath,
                    "num_threads": arguments.threads,
                    "batch_size": arguments.batchsize,
                    "api": arguments.api,
                    "subgraphs": arguments.subgraphs,
                })
                t.start()
                threads.append(t)
                # eval_program(
                #     program_path=progpath,
                #     test_path=arguments.trainpath,
                #     test_limit=arguments.limit,
                #     test_result_path=arguments.outpath,
                #     num_threads=arguments.threads,
                #     batch_size=arguments.batchsize,
                #     api=arguments.api,
                # )
            elif arguments.evalval and progpath is not None:
                t = Process(target=eval_program, kwargs={
                    "program_path": progpath,
                    "test_path": arguments.valpath,
                    "test_limit": arguments.limit,
                    "test_result_path": arguments.outpath,
                    "num_threads": arguments.threads,
                    "batch_size": arguments.batchsize,
                    "api": arguments.api,
                    "subgraphs": arguments.subgraphs,
                })
                t.start()
                threads.append(t)

                # eval_program(
                #     program_path=progpath,
                #     test_path=arguments.valpath,
                #     test_limit=arguments.limit,
                #     test_result_path=arguments.outpath,
                #     num_threads=arguments.threads,
                #     batch_size=arguments.batchsize,
                #     api=arguments.api,
                #     #fewshot=arguments.fewshot,
                # )
            elif arguments.evaltest and progpath is not None:
                t = Process(target=eval_program, kwargs={
                    "program_path": progpath,
                    "test_path": arguments.testpath,
                    "test_limit": arguments.limit,
                    "test_result_path": arguments.outpath,
                    "num_threads": arguments.threads,
                    "batch_size": arguments.batchsize,
                    "api": arguments.api,
                    "subgraphs": arguments.subgraphs,
                })
                t.start()
                threads.append(t)

                # eval_program(
                #     program_path=progpath,
                #     test_path=arguments.testpath,
                #     test_limit=arguments.limit,
                #     test_result_path=arguments.outpath,
                #     num_threads=arguments.threads,
                #     batch_size=arguments.batchsize,
                #     api=arguments.api,
                #     #fewshot=arguments.fewshot,
                # )
            else:
                print("Invalid Arguments")
                argparser.print_help()

    for t in threads:
        t.join()
        print("Thread finished")

    print("All done")

    exit(0)
