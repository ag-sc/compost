import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import traceback
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, Any

import optuna
import torch
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from optuna import Trial
from transformers import AutoTokenizer

from dudes import consts, utils
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from llm.compositionality.dataset import CompDataModule
import llm.compositionality.training as training
from llm.compositionality.models.comp_model import CompModel
from llm.utils import run_nvidia_smi

cache: Dict[str, Dict[str, Any]] = dict()
sparql_endpoint = SPARQLEndpoint(endpoint=consts.sparql_endpoint, cache=cache, check_endpoint=False, default_graph="http://dbpedia.org")

def objective(trial: Trial,
              batch_size=None,
              learning_rate=None,
              ld=None,
              epochs=None,
              model_name=None,
              modelout=None,
              resultout=None,
              tblogs=None,
              datalimit=None,
              instruct=None,
              subgraphs=None,
              endpoint=None,
              trainpath=None,
              valpath=None,
              testpath=None):
    try:
        if batch_size is None:
            batch_size = trial.suggest_int('batch_size', 1, 6)
        if learning_rate is None:
            learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-3, log=True)
        if ld is None:
            ld = trial.suggest_float('ld', 0.8, 1.0, log=True)
        if epochs is None:
            epochs = trial.suggest_int('epochs', 1, 16)
        if model_name is None:
            model_name = "allenai/OLMo-2-1124-7B-Instruct"
        if datalimit is None:
            datalimit = 0
        if instruct is None:
            instruct = True
        if subgraphs is None:
            subgraphs = True

        if trainpath is None:
            if subgraphs:
                trainpath = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "compositionality_subgraph_max_10_fixed.json"
                )
            else:
                trainpath = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "comp",
                    "train-full.csv"
                )
        if valpath is None:
            if subgraphs:
                valpath = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "compositionality_subgraph_max_10_fixed.json"
                )
            else:
                valpath = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "comp",
                    "val-full.csv"
                )
        if testpath is None:
            if subgraphs:
                testpath = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "compositionality_subgraph_max_10_fixed.json"
                )
            else:
                testpath = os.path.join(
                    os.path.dirname(sys.modules["lemon"].__file__),
                    "resources",
                    "comp",
                    "test-full.csv"
                )
        for tryid in range(3):#three tries to train model
            print("Try ID: ", tryid, flush=True)
            cachedir = "./"
            if os.path.isdir("/vol/neodudes/neodudes/"):
                cachedir = "/vol/neodudes/neodudes/"
            elif 'SLURM_JOB_TMP' in os.environ and os.environ['SLURM_JOB_TMP'] is not None and len(os.environ['SLURM_JOB_TMP']) > 0 and os.path.isdir(os.environ['SLURM_JOB_TMP']):
                cachedir = os.environ['SLURM_JOB_TMP']
            elif os.path.isdir("/local/huggingface/custom/"):
                cachedir = "/local/huggingface/custom/"

            print("Cachedir: ", cachedir, flush=True)

            basename = f"comp_llm_{utils.slugify(model_name)}_{learning_rate}_{ld}_{batch_size}_{epochs}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')}"
            tblogs_basename = f"tb_logs_{os.getpid()}"
            basepath = os.path.join(cachedir, basename) #"/local/huggingface/custom/"+basename if os.path.isdir("/local/huggingface/custom/") else basename
            if tblogs is None:
                tblogs_basepath = os.path.join(cachedir, tblogs_basename)
            else:
                tblogs_basepath = tblogs
            if modelout is None:
                modelout = basepath
            if resultout is None:
                resultout = basepath+"_results.json"

            res = subprocess.run([
                "python",
                os.path.abspath(sys.modules["llm.compositionality.training"].__file__),
                "--model",
                model_name,
                "--batchsize",
                str(batch_size),
                "--epochs",
                str(epochs),
                "--lr",
                str(learning_rate),
                "--ld",
                str(ld),
                "--modelout",
                modelout,
                "--resultout",
                resultout,
                "--tblogs",
                tblogs_basepath,
                "--datalimit",
                str(datalimit),
                "--trainpath",
                trainpath,
                "--valpath",
                valpath,
                "--testpath",
                testpath,
            ]+(
                    (["--instruct"] if instruct else [])+
                    (["--subgraphs"] if subgraphs else [])+
                    (["--endpoint", endpoint] if endpoint is not None else [])
            ), stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)

            try:
                with open(resultout, "r") as f:
                    result = json.load(f)

                if float(result["val_loss"]) > 0.9999:
                    rmcmd = [
                        "rm",
                        "-r",
                        basepath+".ckpt",
                    ]
                    print("Validation loss did not improve from 0, removing model", flush=True)
                    print(rmcmd, flush=True)
                    resrm = subprocess.run(rmcmd, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr)

                return result["val_loss"]
            except Exception as e:
                print(traceback.format_exc(), flush=True)
                print("Error loading result", e, flush=True)

        print("Error training model", flush=True)
        raise optuna.TrialPruned()
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        print(e, flush=True)
        raise optuna.TrialPruned()

def evaluate(
        path,
        batch_size,
        val_not_test=False,
        datalimit=0,
        instruct=True,
        subgraphs=True,
        trainpath=None,
        valpath=None,
        testpath=None,
):
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
    dm = CompDataModule(
        tokenizer=AutoTokenizer.from_pretrained(model_name),
        batch_size=batch_size,
        datalimit=datalimit,
        is_instruct=instruct,
        is_subgraph=subgraphs,
        train_path=trainpath,
        val_path=valpath,
        test_path=testpath
    )
    dm.prepare_data()
    dm.setup("fit")
    dm.setup("test")

    model = CompModel(model_name=model_name,
                      learning_rate=learning_rate,
                      ld=ld,
                      sparql_endpoint=sparql_endpoint)#SPARQLEndpoint(endpoint="http://dbpedia.org/sparql", cache=cache))
    #if val_not_test:
    ret_train = trainer.test(model, dataloaders=dm.train_dataloader(), ckpt_path=path)  # , ckpt_path=path)
    print(trainer.callback_metrics, flush=True)
    ret_val = trainer.validate(model, datamodule=dm, ckpt_path=path)
    print(trainer.callback_metrics, flush=True)
    #else:
    ret_test = trainer.test(model, datamodule=dm, ckpt_path=path)#, ckpt_path=path)
    print(trainer.callback_metrics, flush=True)

    return ret_val, trainer.callback_metrics

if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--eval", type=str, default=None)
    argparser.add_argument('--validation', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--instruct', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--subgraphs', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--model", type=str, default="allenai/OLMo-2-1124-7B-Instruct")
    argparser.add_argument("--batchsize", type=int, default=None)
    argparser.add_argument("--epochs", type=int, default=None)
    argparser.add_argument("--lr", type=float, default=None)
    argparser.add_argument("--ld", type=float, default=None)
    argparser.add_argument("--trials", type=int, default=1)
    argparser.add_argument("--modelout", type=str, default=None)
    argparser.add_argument("--resultout", type=str, default=None)
    argparser.add_argument("--tblogs", type=str, default=None)
    argparser.add_argument("--optunafile", type=str, default=f"comp_optuna.log")#_{datetime.now().strftime('%Y-%m-%d')}
    argparser.add_argument("--studyname", type=str, default=f"Compositionality")# {datetime.now().strftime('%Y-%m-%d')}
    argparser.add_argument("--datalimit", type=int, default=0)
    argparser.add_argument("--endpoint", type=str, default=None)
    argparser.add_argument("--trainpath", type=str, default=None)
    argparser.add_argument("--valpath", type=str, default=None)
    argparser.add_argument("--testpath", type=str, default=None)

    arguments = argparser.parse_args()
    print(arguments, flush=True)
    print(datetime.now(), flush=True)

    thread = threading.Thread(target=run_nvidia_smi, daemon=True)
    thread.start()

    if arguments.endpoint is not None:
        sparql_endpoint = SPARQLEndpoint(endpoint=arguments.endpoint, cache=dict(), check_endpoint=False, default_graph="http://dbpedia.org")


    if arguments.eval is not None:
        print("Evaluating model", arguments.eval)
        res = evaluate(
            arguments.eval,
            arguments.batchsize,
            arguments.validation,
            datalimit=arguments.datalimit,
            instruct=arguments.instruct,
            subgraphs=arguments.subgraphs,
            trainpath=arguments.trainpath,
            valpath=arguments.valpath,
            testpath=arguments.testpath,
        )
        print(res)
    else:
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(arguments.optunafile)
        )
        #JournalStorage(JournalFileStorage(f"{arguments.optunafile}"))
        study = optuna.create_study(direction='minimize',
                                    study_name=f"{arguments.studyname}",
                                    storage=storage, load_if_exists=True)
        study.optimize(lambda trial: objective(trial=trial,
                                               batch_size=arguments.batchsize,
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
                                               endpoint=arguments.endpoint,
                                               trainpath=arguments.trainpath,
                                               valpath=arguments.valpath,
                                               testpath=arguments.testpath),
                       n_trials=arguments.trials,
                       n_jobs=1)
        print(study.best_params)

    print("Finished", datetime.now(), flush=True)
