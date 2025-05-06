import argparse
import json
import os.path
import pathlib
import re
import sys
import traceback
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import optuna
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict
from optuna import Trial
from optuna.storages import JournalStorage, JournalFileStorage
from lightning.pytorch.strategies import FSDPStrategy, DeepSpeedStrategy

from dudes import consts
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from llm.compositionality.dataset import CompDataModule
from llm.compositionality.models.comp_model import CompModel

from transformers import AutoTokenizer

cache: Dict[str, Dict[str, Any]] = dict()
sparql_endpoint = SPARQLEndpoint(endpoint=consts.sparql_endpoint, cache=cache, check_endpoint=False, default_graph="http://dbpedia.org")

def objective(batch_size,
              learning_rate,
              ld,
              epochs,
              model_name,
              modelout,
              resultout,
              tblogs,
              datalimit,
              instruct,
              subgraphs,
              trainpath,
              valpath,
              testpath,):
    try:


        model = CompModel(model_name=model_name,
                          learning_rate=learning_rate,
                          ld=ld, sparql_endpoint=None,
                          is_instruct=instruct,
                          is_subgraph=subgraphs)
        model.sparql_endpoint = sparql_endpoint #SPARQLEndpoint(endpoint="http://dbpedia.org/sparql", cache=dict())
        dm = CompDataModule(tokenizer=model.tokenizer,
                            batch_size=batch_size,
                            datalimit=datalimit,
                            is_instruct=instruct,
                            is_subgraph=subgraphs,
                            train_path=trainpath,
                            val_path=valpath,
                            test_path=testpath)
        dm.prepare_data()
        dm.setup("train")

        print("Batch size: ", batch_size, flush=True)
        print("Learning rate: ", learning_rate, flush=True)
        print("Lambda: ", ld, flush=True)
        print("Epochs: ", epochs, flush=True)
        print("Model: ", model_name, flush=True)

        logger = TensorBoardLogger(tblogs,#"/local/huggingface/custom/tb_logs_"+str(os.getpid()) if os.path.isdir("/local/huggingface/custom") else "tb_logs",
                                   name="comp_llm",
                                   version=None if 'SLURM_ARRAY_TASK_ID' not in os.environ else os.environ['SLURM_ARRAY_TASK_ID'])

        trainer = Trainer(enable_checkpointing=False,
                          logger=logger,
                          #accelerator="cpu",
                          reload_dataloaders_every_n_epochs=1,
                          log_every_n_steps=1,
                          max_epochs=epochs,
                          min_epochs=epochs,
                          accelerator="cuda",
                          devices=-1,
                          strategy="deepspeed_stage_2_offload",
                          # strategy=DeepSpeedStrategy(
                          #     config=os.path.join(
                          #         os.path.dirname(sys.modules["lemon"].__file__),
                          #         "resources",
                          #         "deepspeed.json"
                          #     ),
                          # ),
                          #precision="16-mixed",
                          precision="bf16-mixed",
                          #precision="16-mixed",
                          #plugins=BitsandbytesPrecision(mode="int8"),
                          #plugins=BitsandbytesPrecision(mode="int8-training", dtype=torch.float16, ignore_modules={"lm_head"}),
                          #strategy="fsdp"
                          # strategy=FSDPStrategy(activation_checkpointing_policy={
                          #   torch.nn.TransformerEncoderLayer,
                          #   torch.nn.TransformerDecoderLayer,
                          # }, cpu_offload=True),
                          #accumulate_grad_batches=8,
                          )
        trainer.fit(model, datamodule=dm)

        trainer.save_checkpoint(modelout+".ckpt")
        print("Saved model to", modelout+".ckpt", flush=True)
        #convert_zero_checkpoint_to_fp32_state_dict(modelout+".ckpt", modelout+".pt")
        #print("Converted model to", modelout+".pt", flush=True)

        print(trainer.callback_metrics, flush=True)
        ret_val = trainer.callback_metrics["val_loss"]

        try:
            train_loss = float(trainer.callback_metrics["train_loss"]) if "train_loss" in trainer.callback_metrics else None
        except Exception as e:
            train_loss = None
            print(e, flush=True)

        try:
            val_loss = float(trainer.callback_metrics["val_loss"]) if "val_loss" in trainer.callback_metrics else None
        except Exception as e:
            val_loss = None
            print(e, flush=True)

        trainer.test(model, datamodule=dm)
        print(trainer.callback_metrics, flush=True)

        try:
            test_loss = float(trainer.callback_metrics["test_loss"]) if "test_loss" in trainer.callback_metrics else None
        except Exception as e:
            test_loss = None
            print(e, flush=True)

        try:
            with open(resultout, "w") as f:
                json.dump({"train_loss": train_loss, "val_loss": val_loss, "test_loss": test_loss}, f)
        except:
            print("Could not write to", resultout, flush=True)

        return ret_val
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        print(e, flush=True)
        raise e



if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('--validation', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--instruct', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--subgraphs', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--model", type=str, default="allenai/OLMo-2-1124-7B")
    argparser.add_argument("--batchsize", type=int, default=1)
    argparser.add_argument("--epochs", type=int)
    argparser.add_argument("--lr", type=float)
    argparser.add_argument("--ld", type=float)
    argparser.add_argument("--modelout", type=str)
    argparser.add_argument("--resultout", type=str)
    argparser.add_argument("--tblogs", type=str)
    argparser.add_argument("--datalimit", type=int, default=0)
    argparser.add_argument("--endpoint", type=str, default=None)
    argparser.add_argument("--trainpath", type=str, default=None)
    argparser.add_argument("--valpath", type=str, default=None)
    argparser.add_argument("--testpath", type=str, default=None)

    arguments = argparser.parse_args()

    if arguments.endpoint is not None:
        sparql_endpoint = SPARQLEndpoint(endpoint=arguments.endpoint, cache=dict(), check_endpoint=False, default_graph="http://dbpedia.org")

    objective(batch_size=arguments.batchsize,
              epochs=arguments.epochs,
              model_name=arguments.model,
              learning_rate=arguments.lr,
              ld=arguments.ld,
              modelout=arguments.modelout,
              resultout=arguments.resultout,
              tblogs=arguments.tblogs,
              datalimit=arguments.datalimit,
              instruct=arguments.instruct,
              subgraphs=arguments.subgraphs,
              trainpath=arguments.trainpath,
              valpath=arguments.valpath,
              testpath=arguments.testpath)
