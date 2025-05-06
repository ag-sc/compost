import csv
import io
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# cache_path = Path(f"/dev/shm/{os.getpid()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
# cache_path.mkdir(parents=True, exist_ok=True)
#
# os.environ["DSPY_CACHEDIR"] = str(cache_path)
# os.environ["DSP_CACHEDIR"] = str(cache_path)
# import dspy  # type: ignore

import csv
import os
import sys
from typing import List, Optional, Dict

import lightning as L
import torch
import torch.nn.functional as F
import transformers
import zstandard

import lemon
import random

from torch.utils.data import random_split, DataLoader, Dataset

from dudes import consts


class CompDataset(Dataset):
    def __init__(self, data: List, pad_token_id: int, pad_multiplier: int = 512):
        self.data = data
        self.pad_token_id = pad_token_id
        self.pad_multiplier = pad_multiplier
        input_max_length = max([len(de["input"]) for de in data])
        output_max_length = max([len(de["target"]) for de in data])
        self.input_target_length = input_max_length#max(input_max_length, output_max_length)#512 * (input_max_length // self.pad_multiplier + 1)
        self.output_target_length = output_max_length#max(input_max_length, output_max_length) #512 * (output_max_length // self.pad_multiplier + 1)
        #raw_input_max_length = max([len(list(bytes(de["raw_input"], 'utf8'))) for de in data])
        #self.input_target_length_raw = max([len(list(bytes(de["raw_input"], 'utf8'))) for de in data]) #512 * (raw_input_max_length // self.pad_multiplier + 1)
        #raw_output_max_length = max([len(list(bytes(de["raw_target"], 'utf8'))) for de in data])
        #self.output_target_length_raw = max([len(list(bytes(de["raw_target"], 'utf8'))) for de in data]) #512 * (raw_output_max_length // self.pad_multiplier + 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        de: Dict[str, str] = self.data[idx]
        tensor_input: torch.Tensor = torch.LongTensor(de["input"])
        tensor_input = F.pad(tensor_input, (self.input_target_length - len(de["input"]), 0), value=self.pad_token_id)#LEFT-pad content for decoder-only models
        tensor_output: torch.Tensor = torch.LongTensor(de["target"])
        tensor_output = F.pad(tensor_output, (0, self.output_target_length - len(de["target"])), value=self.pad_token_id)#right-pad with pad as well, we finetune, should be ok
        #tensor_output = F.pad(tensor_output, (0, self.output_target_length - len(de["target"])), value=-100)#crossentropy ignore index
        #raw_input: torch.Tensor = torch.ByteTensor(list(bytes(de["raw_input"], 'utf8')))
        #raw_input = F.pad(raw_input, (0, self.input_target_length_raw - len(raw_input)), value=32)#32 is space in ASCII/UTF-8
        #raw_target: torch.Tensor = torch.ByteTensor(list(bytes(de["raw_target"], 'utf8')))
        #raw_target = F.pad(raw_target, (0, self.output_target_length_raw - len(raw_target)), value=32)

        return {
            "input_token_ids": tensor_input,
            "output_token_ids": tensor_output,
            "id": de["id"],
            "subid": de["subid"],
            #"raw_input": raw_input,
            #"raw_target": raw_target,
        }


class CompDataModule(L.LightningDataModule):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 train_path: Optional[str] = None,
                 val_path: Optional[str] = None,
                 test_path: Optional[str] = None,
                 batch_size: int = 32,
                 datalimit: int = 0,
                 is_instruct: bool = False,
                 is_subgraph: bool = True,):
        super().__init__()
        self.train: Optional[Dataset] = None
        self.val: Optional[Dataset] = None
        self.test: Optional[Dataset] = None

        if train_path is None:
            if is_subgraph:
                self.train_path = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "compositionality_subgraph_max_10_fixed.json"
                )
            else:
                self.train_path = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "comp",
                    "train-full.csv"
                )
        else:
            self.train_path = train_path
        if val_path is None:
            if is_subgraph:
                self.val_path = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "compositionality_subgraph_max_10_fixed.json"
                )
            else:
                self.val_path = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "comp",
                    "val-full.csv"
                )
        else:
            self.val_path = val_path
        if test_path is None:
            if is_subgraph:
                self.test_path = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "compositionality_subgraph_max_10_fixed.json"
                )
            else:
                self.test_path = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "comp",
                    "test-full.csv"
                )
        else:
            self.test_path = test_path

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.datalimit = datalimit
        self.is_instruct = is_instruct
        self.is_subgraph = is_subgraph

    def prepare_data(self):
        pass

    def _load_data(self, path, skip_third=False):
        with open(path) as csv_file:
            csv_dict = csv.DictReader(csv_file, delimiter=',')
            if self.is_instruct:
                data_raw = sum([
                    [
                        {
                            "input": self.tokenizer.apply_chat_template([
                                {"role": "user", "content": de["question1"]},
                            ], tokenize=True, add_generation_prompt=True),
                            "target": self.tokenizer.apply_chat_template([
                                {"role": "user", "content": de["question1"]},
                                {"role": "assistant", "content": de["query1"]},
                            ], tokenize=True, add_generation_prompt=False),
                            "id": i,
                            "subid": 1,
                            # "raw_input": de["question1"],
                            # "raw_target": de["query1"],
                        },
                        {
                            "input": self.tokenizer.apply_chat_template([
                                {"role": "user", "content": de["question2"]},
                            ], tokenize=True, add_generation_prompt=True),
                            "target": self.tokenizer.apply_chat_template([
                                {"role": "user", "content": de["question2"]},
                                {"role": "assistant", "content": de["query2"]},
                            ], tokenize=True, add_generation_prompt=False),
                            "id": i,
                            "subid": 2,
                            # "raw_input": de["question2"],
                            # "raw_target": de["query2"],
                        },
                    ] + ([
                             {
                                 "input": self.tokenizer.apply_chat_template([
                                     {"role": "user", "content": de["question3"]},
                                 ], tokenize=True, add_generation_prompt=True),
                                 "target": self.tokenizer.apply_chat_template([
                                     {"role": "user", "content": de["question3"]},
                                     {"role": "assistant", "content": de["query3"]},
                                 ], tokenize=True, add_generation_prompt=False),
                                 "id": i,
                                 "subid": 3,
                                 # "raw_input": de["question3"],
                                 # "raw_target": de["query3"],
                             }
                         ] if not skip_third else [])
                    for i, de in enumerate(csv_dict) if self.datalimit == 0 or i < self.datalimit
                ], [])
            else:
                data_raw = sum([
                    [
                        {
                            "input": self.tokenizer.encode(de["question1"] + " SPARQL Query: "),
                            "target": self.tokenizer.encode(de["query1"]) + [self.tokenizer.eos_token_id],
                            "id": i,
                            "subid": 1,
                            #"raw_input": de["question1"],
                            #"raw_target": de["query1"],
                        },
                        {
                            "input": self.tokenizer.encode(de["question2"] + " SPARQL Query: "),
                            "target": self.tokenizer.encode(de["query2"]) + [self.tokenizer.eos_token_id],
                            "id": i,
                            "subid": 2,
                            #"raw_input": de["question2"],
                            #"raw_target": de["query2"],
                        },
                    ] + ([
                        {
                            "input": self.tokenizer.encode(de["question3"] + " SPARQL Query: "),
                            "target": self.tokenizer.encode(de["query3"]) + [self.tokenizer.eos_token_id],
                            "id": i,
                            "subid": 3,
                            #"raw_input": de["question3"],
                            #"raw_target": de["query3"],
                        }
                    ] if not skip_third else [])
                    for i, de in enumerate(csv_dict) if self.datalimit == 0 or i < self.datalimit
                    ], [])
            return data_raw

    def _load_dataset(self, path, skip_third=False):
        data_raw = self._load_data(path, skip_third=skip_third)
        return CompDataset(data_raw, self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)

    def _load_data_subgraph(self, path, dataset="train"):
        with open(path) as json_file:
            if path.endswith(".zst"):
                dctx = zstandard.ZstdDecompressor()
                stream_reader = dctx.stream_reader(json_file)
                json_file = io.TextIOWrapper(stream_reader, encoding='utf-8')
            data_json = json.load(json_file)
            if self.is_instruct:
                data_raw = [
                    {
                        "input": self.tokenizer.apply_chat_template([
                            {"role": "system", "content": consts.openai_subgraph_prompt},
                            {"role": "user", "content": de["question"]},
                        ], tokenize=True, add_generation_prompt=True),
                        "target": self.tokenizer.apply_chat_template([
                            {"role": "system", "content": consts.openai_subgraph_prompt},
                            {"role": "user", "content": de["question"]},
                            {"role": "assistant", "content": de["query"]},
                        ], tokenize=True, add_generation_prompt=False),
                        "id": de["id"],
                        "subid": de["gid"],
                        # "raw_input": de["question1"],
                        # "raw_target": de["query1"],
                    }
                    for i, de in enumerate(data_json[dataset]) if self.datalimit == 0 or i < self.datalimit
                ]
            else:
                data_raw = [
                    {
                        "input": self.tokenizer.encode(consts.openai_subgraph_prompt + " Question: " + de["question"] + " SPARQL Query: "),
                        "target": self.tokenizer.encode(de["query"]) + [self.tokenizer.eos_token_id],
                        "id": de["id"],
                        "subid": de["gid"],
                        # "raw_input": de["question1"],
                        # "raw_target": de["query1"],
                    }
                    for i, de in enumerate(data_json[dataset]) if self.datalimit == 0 or i < self.datalimit
                ]

            print(f"Dataset sizes: {dataset}={len(data_raw)}", flush=True)
            return data_raw

    def _load_dataset_subgraph(self, path, dataset="train"):
        data_raw = self._load_data_subgraph(path, dataset=dataset)
        return CompDataset(data_raw, self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if self.is_subgraph:
            if stage == "fit" or stage is None:
                full_data = self._load_data_subgraph(self.train_path, dataset="train")
                # full_data += self._load_data(self.val_path, skip_third=True)
                # full_data += self._load_data(self.test_path, skip_third=True)

                self.train = CompDataset(full_data,
                                         self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)  # self._load_dataset(self.train_path)
                self.val = self._load_dataset_subgraph(self.val_path, dataset="val")

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage == "predict":
                self.test = self._load_dataset_subgraph(self.test_path, dataset="test")
        else:
            if stage == "fit" or stage is None:
                full_data = self._load_data(self.train_path)
                #full_data += self._load_data(self.val_path, skip_third=True)
                #full_data += self._load_data(self.test_path, skip_third=True)

                self.train = CompDataset(full_data, self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)#self._load_dataset(self.train_path)
                self.val = self._load_dataset(self.val_path)

            # Assign test dataset for use in dataloader(s)
            if stage == "test" or stage == "predict":
                self.test = self._load_dataset(self.test_path)


    def train_dataloader(self):
        assert self.train is not None
        return DataLoader(self.train, shuffle=True, batch_size=self.batch_size, num_workers=29, pin_memory=True)

    def val_dataloader(self):
        assert self.val is not None
        return DataLoader(self.val, shuffle=False, batch_size=self.batch_size, num_workers=29, pin_memory=True)

    def test_dataloader(self):
        assert self.test is not None
        return DataLoader(self.test, shuffle=False, batch_size=self.batch_size, num_workers=29, pin_memory=True)

    def predict_dataloader(self):
        assert self.test is not None
        return DataLoader(self.test, shuffle=False, batch_size=self.batch_size, num_workers=29, pin_memory=True)


def load_dataset(path, limit=None, fixed_fewshot=False, subgraphs=None):
    data = []
    import dspy

    mode = "r"
    if subgraphs is not None and path.endswith(".zst"):
        mode = "rb"
    with open(path, mode) as file:
        if subgraphs is not None:
            if path.endswith(".zst"):
                dctx = zstandard.ZstdDecompressor()
                stream_reader = dctx.stream_reader(file)
                file = io.TextIOWrapper(stream_reader, encoding='utf-8')
            if not fixed_fewshot:
                inputs = ["question"]
                jsondata = json.load(file)
                rnd = random.Random(42)
                rnd.shuffle(jsondata["train"])
                rnd.shuffle(jsondata["val"])
                rnd.shuffle(jsondata["test"])

                if subgraphs.lower() == "train":
                    data = [
                        dspy.Example(
                            question=t["question"],
                            sparql_query=t["query"],
                            id=t["id"],
                            subid=t["gid"],
                            split="train",
                            num_edges=t["num_edges"],
                            num_edges_full=t["num_edges_full"] if "num_edges_full" in t else t["num_edges"],
                            num_nodes=t["num_nodes"],
                            depth=t["depth"],
                            breadth=t["breadth"],
                            base_depth=t["base_depth"],
                            base_breadth=t["base_breadth"],
                            subgraphs=t["subgraphs"],
                        )
                        for i, t in enumerate(jsondata["train"]) if limit is None or i < limit
                    ]
                elif subgraphs.lower() == "val":
                    data = [
                        dspy.Example(
                            question=t["question"],
                            sparql_query=t["query"],
                            id=t["id"],
                            subid=t["gid"],
                            split="val",
                            num_edges=t["num_edges"],
                            num_edges_full=t["num_edges_full"] if "num_edges_full" in t else t["num_edges"],
                            num_nodes=t["num_nodes"],
                            depth=t["depth"],
                            breadth=t["breadth"],
                            base_depth=t["base_depth"],
                            base_breadth=t["base_breadth"],
                            subgraphs=t["subgraphs"],
                        )
                        for i, t in enumerate(jsondata["val"]) if limit is None or i < limit
                    ]
                elif subgraphs.lower() == "test":
                    data = [
                        dspy.Example(
                            question=t["question"],
                            sparql_query=t["query"],
                            id=t["id"],
                            subid=t["gid"],
                            split="test",
                            num_edges=t["num_edges"],
                            num_edges_full=t["num_edges_full"] if "num_edges_full" in t else t["num_edges"],
                            num_nodes=t["num_nodes"],
                            depth=t["depth"],
                            breadth=t["breadth"],
                            base_depth=t["base_depth"],
                            base_breadth=t["base_breadth"],
                            subgraphs=t["subgraphs"],
                        )
                        for i, t in enumerate(jsondata["test"]) if limit is None or i < limit
                    ]
            else:
                inputs = ["question", "shots"]
                jsondata = json.load(file)["fixed_shots"]
                rnd = random.Random(42)
                rnd.shuffle(jsondata["train"])
                rnd.shuffle(jsondata["test"])

                if subgraphs.lower() == "train" or subgraphs.lower() == "val":
                    if limit is None or len(jsondata["train"]) < 2 * limit:
                        data = [
                            dspy.Example(
                                question=t["task"]["question"],
                                sparql_query=t["task"]["query"],
                                shots=[(s["question"], s["query"]) for s in t["shots"]],
                                id=t["task"]["id"],
                                subid=t["task"]["gid"],
                                split="train",
                                num_edges=t["task"]["num_edges"],
                                num_edges_full=t["task"]["num_edges_full"] if "num_edges_full" in t["task"] else t["task"]["num_edges"],
                                num_nodes=t["task"]["num_nodes"],
                                depth=t["task"]["depth"],
                                breadth=t["task"]["breadth"],
                                base_depth=t["task"]["base_depth"],
                                base_breadth=t["task"]["base_breadth"],
                                subgraphs=t["task"]["subgraphs"],
                            )
                            for i, t in enumerate(jsondata["train"]) if limit is None or i < limit
                        ]
                    else:
                        # if len(jsondata["train"]) >= 2*limit:
                        trainval = rnd.sample(jsondata["train"], 2 * limit)
                        train = trainval[:limit]
                        val = trainval[limit:]
                        if subgraphs.lower() == "train":
                            data = [
                                dspy.Example(
                                    question=t["task"]["question"],
                                    sparql_query=t["task"]["query"],
                                    shots=[(s["question"], s["query"]) for s in t["shots"]],
                                    id=t["task"]["id"],
                                    subid=t["task"]["gid"],
                                    split="train",
                                    num_edges=t["task"]["num_edges"],
                                    num_edges_full=t["task"]["num_edges_full"] if "num_edges_full" in t["task"] else t["task"]["num_edges"],
                                    num_nodes=t["task"]["num_nodes"],
                                    depth=t["task"]["depth"],
                                    breadth=t["task"]["breadth"],
                                    base_depth=t["task"]["base_depth"],
                                    base_breadth=t["task"]["base_breadth"],
                                    subgraphs=t["task"]["subgraphs"],
                                )
                                for i, t in enumerate(train)
                            ]
                        else:
                            data = [
                                dspy.Example(
                                    question=t["task"]["question"],
                                    sparql_query=t["task"]["query"],
                                    shots=[(s["question"], s["query"]) for s in t["shots"]],
                                    id=t["task"]["id"],
                                    subid=t["task"]["gid"],
                                    split="val",
                                    num_edges=t["task"]["num_edges"],
                                    num_edges_full=t["task"]["num_edges_full"] if "num_edges_full" in t["task"] else t["task"]["num_edges"],
                                    num_nodes=t["task"]["num_nodes"],
                                    depth=t["task"]["depth"],
                                    breadth=t["task"]["breadth"],
                                    base_depth=t["task"]["base_depth"],
                                    base_breadth=t["task"]["base_breadth"],
                                    subgraphs=t["task"]["subgraphs"],
                                )
                                for i, t in enumerate(val)
                            ]
                elif subgraphs.lower() == "test":
                    data = [
                        dspy.Example(
                            question=t["task"]["question"],
                            sparql_query=t["task"]["query"],
                            shots=[(s["question"], s["query"]) for s in t["shots"]],
                            id=t["task"]["id"],
                            subid=t["task"]["gid"],
                            split="test",
                            num_edges=t["task"]["num_edges"],
                            num_edges_full=t["task"]["num_edges_full"] if "num_edges_full" in t["task"] else t["task"]["num_edges"],
                            num_nodes=t["task"]["num_nodes"],
                            depth=t["task"]["depth"],
                            breadth=t["task"]["breadth"],
                            base_depth=t["task"]["base_depth"],
                            base_breadth=t["task"]["base_breadth"],
                            subgraphs=t["task"]["subgraphs"],
                        )
                        for i, t in enumerate(jsondata["test"]) if limit is None or i < limit
                    ]
        else:
            if fixed_fewshot:
                inputs = ["shot1_question", "shot1_sparql_query", "shot2_question", "shot2_sparql_query", "question"]
                data = [
                    dspy.Example(
                        question=d["question3"],
                        sparql_query=d["query3"],
                        shot1_question=d["question1"],
                        shot1_sparql_query=d["query1"],
                        shot2_question=d["question2"],
                        shot2_sparql_query=d["query2"],
                        id = i,
                        subid = None,
                    )
                    for i, d in enumerate(csv.DictReader(file, delimiter=',')) if limit is None or i < limit
                ]
            else:
                inputs = ["question"]
                data = sum([
                    [
                        dspy.Example(question=d["question1"], sparql_query=d["query1"], id=i, subid=1),
                        dspy.Example(question=d["question2"], sparql_query=d["query2"], id=i, subid=2),
                        dspy.Example(question=d["question3"], sparql_query=d["query3"], id=i, subid=3),
                    ]
                    for i, d in enumerate(csv.DictReader(file, delimiter=',')) if limit is None or i < limit
                ], [])

    data = [d.with_inputs(*inputs) for d in data]


    return data


def load_datasets(
        train_path=None,
        val_path=None,
        test_path=None,
        train_limit=None,
        val_limit=None,
        test_limit=None,
        fixed_fewshot=False,
        subgraphs=False
):
    if subgraphs:
        paths = [p for p in [train_path, val_path, test_path] if p is not None]

        if len(paths) == 0:
            if fixed_fewshot:
                paths = [os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "compositionality_subgraph_hard_prompt_5.json.zst"
                )]
            else:
                paths = [os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "compositionality_subgraph_hard.json"
                )]

        path = paths[0]

        logging.debug(f"Loading dataset from {path}")

        train = load_dataset(path, limit=train_limit, fixed_fewshot=fixed_fewshot, subgraphs="train")
        val = load_dataset(path, limit=val_limit, fixed_fewshot=fixed_fewshot, subgraphs="val")
        test = load_dataset(path, limit=test_limit, fixed_fewshot=fixed_fewshot, subgraphs="test")

        logging.debug(f"Dataset sizes: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test
    else:
        if train_path is None:
            train_path = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources",
                "comp",
                "train-full.csv"
            )
        if val_path is None:
            val_path = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources",
                "comp",
                "val-full.csv"
            )
        if test_path is None:
            test_path = os.path.join(
                os.path.dirname(sys.modules["lemon"].__file__),
                "resources",
                "comp",
                "test-full.csv"
            )

        train = load_dataset(train_path, limit=train_limit, fixed_fewshot=fixed_fewshot)
        val = load_dataset(val_path, limit=val_limit, fixed_fewshot=fixed_fewshot)
        test = load_dataset(test_path, limit=test_limit, fixed_fewshot=fixed_fewshot)
        return train, val, test

