import argparse
import csv
import io
import os
import threading
from argparse import ArgumentParser
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from typing import Literal

from dudes import consts
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from llm.compositionality import evaluation
#from llm.compositionality.prompting_modules import ZeroShotPipeline

# cache_path = Path(f"/dev/shm/{os.getpid()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
# cache_path.mkdir(parents=True, exist_ok=True)
#
# os.environ["DSPY_CACHEDIR"] = str(cache_path)
# os.environ["DSP_CACHEDIR"] = str(cache_path)
# import dspy  # type: ignore
import more_itertools
#from dspy import MIPROv2

from dudes.utils import slugify
from llm.compositionality.dataset import load_datasets

from llm.compositionality.evaluation import validate_context_and_answer
from llm.utils import configure_dspy, run_nvidia_smi


def zeroshot_plain(
        model="ollama_chat/llama3.3",
        predictor_module=None,
        api=None,
        dry_run=False,
        rootpath=None,
        subgraphs=False
):
    basename_untimestamped = f"zeroshot_plain_{slugify(predictor_module)}_{slugify(model.split('/')[-1])}"
    try:
        basepath = configure_dspy(model, basename_untimestamped, dry_run=dry_run, api=api, rootpath=rootpath)
    except ValueError as e:
        print(e)
        return

    from llm.compositionality.prompting_modules import ZeroShotPipeline
    baseline = ZeroShotPipeline(predictor_module=predictor_module)
    print(baseline)

    baseline.save(basepath, save_program=True)
    baseline.save(str(Path(basepath).with_suffix('.json')), save_program=False)
    print("Saved to", basepath)


def zeroshot_copro(
        train_path=None,
        val_path=None,
        #test_path=None,
        model="ollama_chat/llama3.3",
        num_threads=1,
        train_limit=None,
        val_limit=None,
        #test_limit=None,
        depth=3,
        breadth=10,
        predictor_module=None,
        api=None,
        dry_run=False,
        rootpath=None,
        subgraphs=False,
):
    basename_untimestamped = f"zeroshot_copro_{slugify(predictor_module)}_{slugify(model.split('/')[-1])}_{train_limit}_{val_limit}_{depth}_{breadth}"
    try:
        basepath = configure_dspy(model, basename_untimestamped, dry_run=dry_run, api=api, rootpath=rootpath)
    except ValueError as e:
        print(e)
        return

    train, val, test = load_datasets(
        train_path=train_path,
        val_path=val_path,
        #test_path=test_path,
        train_limit=train_limit,
        val_limit=val_limit,
        #test_limit=test_limit,
        fixed_fewshot=False,
        subgraphs=subgraphs
    )

    # evaluate = Evaluate(
    #     devset=val,
    #     metric=validate_context_and_answer,
    #     num_threads=num_threads,
    #     display_progress=True,
    #     display_table=True,
    #     provide_traceback=True,
    #     return_all_scores=True,
    # )

    from llm.compositionality.prompting_modules import ZeroShotPipeline
    baseline = ZeroShotPipeline(predictor_module=predictor_module)
    print(baseline)

    #evaluate(baseline, devset=val)#[v.with_inputs("question") for v in val]

    import dspy
    teleprompter = dspy.COPRO(
        metric=validate_context_and_answer,
        verbose=True,
        breadth=breadth,
        depth=depth,
    )

    compiled_prompt_opt = teleprompter.compile(baseline,
                                               trainset=train,#[v.with_inputs("question") for v in train],
                                               eval_kwargs={
                                                   'num_threads': num_threads,
                                                   'display_progress': True,
                                                   'display_table': False
                                               })
    print(compiled_prompt_opt)
    compiled_prompt_opt.save(basepath, save_program=True)
    compiled_prompt_opt.save(str(Path(basepath).with_suffix('.json')), save_program=False)
    print("Saved to", basepath)

def zeroshot_mipro(
        train_path=None,
        val_path=None,
        model="ollama_chat/llama3.3",
        train_limit=None,
        val_limit=None,
        num_trials=30,
        minibatch_size=25,
        minibatch_full_eval_steps=10,
        view_data_batch_size=10,
        predictor_module=None,
        auto: Literal["light", "medium", "heavy"]="light",
        api=None,
        dry_run=False,
        rootpath=None,
        subgraphs=False,
):
    assert auto in ["light", "medium", "heavy"]
    basename_untimestamped = f"zeroshot_mipro_{auto}_{slugify(predictor_module)}_{slugify(model.split('/')[-1])}_{train_limit}_{val_limit}_{num_trials}_{minibatch_size}_{minibatch_full_eval_steps}_{view_data_batch_size}"
    try:
        basepath = configure_dspy(model, basename_untimestamped, dry_run=dry_run, api=api, rootpath=rootpath)
    except ValueError as e:
        print(e)
        return

    train, val, test = load_datasets(
        train_path=train_path,
        val_path=val_path,
        train_limit=train_limit,
        val_limit=val_limit,
        fixed_fewshot=False,
        subgraphs=subgraphs
    )

    from llm.compositionality.prompting_modules import ZeroShotPipeline
    baseline = ZeroShotPipeline(predictor_module=predictor_module)
    print(baseline)

    # Initialize optimizer
    from dspy import MIPROv2
    teleprompter = MIPROv2(
        metric=validate_context_and_answer,
        auto=auto,  # Can choose between light, medium, and heavy optimization runs
        max_errors=2*len(train), # ignore errors
        verbose=True
    )

    # Optimize program
    print(f"Optimizing program with MIPRO...")
    optimized_program = teleprompter.compile(
        baseline.deepcopy(),
        trainset=train,
        max_bootstrapped_demos=0,
        max_labeled_demos=0,
        requires_permission_to_run=False,
        num_trials=num_trials,
        minibatch_size=minibatch_size,
        minibatch_full_eval_steps=minibatch_full_eval_steps,
        view_data_batch_size=view_data_batch_size,
    )

    print(optimized_program)

    optimized_program.save(basepath, save_program=True)
    optimized_program.save(str(Path(basepath).with_suffix('.json')), save_program=False)
    print("Saved to", basepath)






if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument("--model", type=str, default=None)
    argparser.add_argument("--threads", type=int, default=8)
    argparser.add_argument("--trainlimit", type=int, default=None)
    argparser.add_argument("--vallimit", type=int, default=None)
    #argparser.add_argument("--testlimit", type=int, default=None)
    argparser.add_argument("--trainpath", type=str, default=None)
    argparser.add_argument("--valpath", type=str, default=None)
    #argparser.add_argument("--testpath", type=str, default=None)
    argparser.add_argument('--plain', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--copro', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--mipro', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--copro-depth", type=int, default=3)
    argparser.add_argument("--copro-breadth", type=int, default=10)
    argparser.add_argument("--mipro-numtrials", type=int, default=30)
    argparser.add_argument("--mipro-minibatchsize", type=int, default=25)
    argparser.add_argument("--mipro-minibatchfullevalsteps", type=int, default=10)
    argparser.add_argument("--mipro-viewdatabatchsize", type=int, default=10)
    argparser.add_argument("--mipro-auto", type=str, default="light", choices=["light", "medium", "heavy"])
    argparser.add_argument("--predictormodule", type=str, default=None)
    argparser.add_argument("--api", type=str, default=None)
    argparser.add_argument("--endpoint", type=str, default=None)

    argparser.add_argument('--autoexp', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--autoexpcsv", type=str, default=None)
    argparser.add_argument("--autoexpcsvid", type=int, default=None)
    argparser.add_argument("--rootpath", type=str, default=None)
    #argparser.add_argument("--batch-frac", type=float, default=1.0)
    #argparser.add_argument("--batch-id", type=int, default=0)
    argparser.add_argument('--dry-run', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--subgraphs', action=argparse.BooleanOptionalAction)

    arguments = argparser.parse_args()
    #print(arguments)

    if arguments.endpoint is not None:
        consts.endpoint = arguments.endpoint
        evaluation.sparql_endpoint = SPARQLEndpoint(endpoint=arguments.endpoint, cache=dict(), check_endpoint=False, default_graph="http://dbpedia.org")

    #assert arguments.batch_frac > 0.0 and arguments.batch_frac <= 1.0
    #assert arguments.batch_id >= 0

    threads = []

    if arguments.autoexp or arguments.autoexpcsv is not None:
        if arguments.autoexp:
            experiments = list(set([
                (func, model, predictor_module, auto if func == "mipro" else None)
                for func in ["plain", "copro", "mipro"]
                for model in (["ollama_chat/llama3.3", "ollama_chat/phi4", "ollama_chat/olmo2:7b-1124-instruct-q4_K_M", "ollama_chat/qwen2.5-coder", "openai/gpt-4o-mini-2024-07-18"] if arguments.model is None else [arguments.model])#, "openai/gpt-4o-mini", "openai/deepseek-r1"
                for predictor_module in ["chainofthought", None]
                for auto in ["light", "medium", "heavy"]
                if not ("deepseek" in model and func == "copro")
            ]))
            #experiments.sort(key=lambda x: str(x))
            experiments.sort(key=lambda x: str(x[1:2] + x[0:1] + x[2:]))

            output = io.StringIO()
            csv_writer = csv.writer(output)
            csv_writer.writerows(experiments)
            csv_string = output.getvalue()
            print(csv_string)
            output.close()
            exit(0)
        # print("Total Experiments", experiments)
        #
        #
        # plain_exps = [(func, model, predictor_module, auto) for func, model, predictor_module, auto in experiments if func == "plain"]
        # nonplain_exps = [(func, model, predictor_module, auto) for func, model, predictor_module, auto in experiments if func != "plain"]
        # nonplain_exps.sort(key=lambda x: str(x))
        #
        # experiments = list(more_itertools.chunked(nonplain_exps, int(len(nonplain_exps)*arguments.batch_frac) + 1))[arguments.batch_id]
        #
        # if arguments.batch_id == 0:
        #     experiments += plain_exps
        #
        # print("Running Experiments", experiments)

        if arguments.autoexpcsv is not None:
            with open(arguments.autoexpcsv) as f:
                experiments = [tuple(line) for idx, line in enumerate(csv.reader(f)) if arguments.autoexpcsvid is None or idx == arguments.autoexpcsvid]

        thread = threading.Thread(target=run_nvidia_smi, daemon=True)
        thread.start()

        for func, model, predictor_module, auto in experiments:
            if func == "plain":
                t = Process(target=zeroshot_plain, kwargs={
                    "model": model,
                    "predictor_module": predictor_module,
                    "api": arguments.api,# if "ollama_chat" in model else None,
                    "dry_run": arguments.dry_run,
                    "rootpath": arguments.rootpath,
                    "subgraphs": arguments.subgraphs
                })
                t.start()
                threads.append(t)
            elif func == "copro":
                t = Process(target=zeroshot_copro, kwargs={
                    "model": model,
                    "predictor_module": predictor_module,
                    "api": arguments.api,# if "ollama_chat" in model else None,
                    "train_path": arguments.trainpath,
                    "val_path": arguments.valpath,
                    "train_limit": arguments.trainlimit,
                    "val_limit": arguments.vallimit,
                    "depth": arguments.copro_depth,
                    "breadth": arguments.copro_breadth,
                    "num_threads": arguments.threads,
                    "dry_run": arguments.dry_run,
                    "rootpath": arguments.rootpath,
                    "subgraphs": arguments.subgraphs
                })
                t.start()
                threads.append(t)
            elif func == "mipro":
                t = Process(target=zeroshot_mipro, kwargs={
                    "model": model,
                    "predictor_module": predictor_module,
                    "api": arguments.api,# if "ollama_chat" in model else None,
                    "train_path": arguments.trainpath,
                    "val_path": arguments.valpath,
                    "train_limit": arguments.trainlimit,
                    "val_limit": arguments.vallimit,
                    "num_trials": arguments.mipro_numtrials,
                    "minibatch_size": arguments.mipro_minibatchsize,
                    "minibatch_full_eval_steps": arguments.mipro_minibatchfullevalsteps,
                    "view_data_batch_size": arguments.mipro_viewdatabatchsize,
                    "auto": auto,
                    "dry_run": arguments.dry_run,
                    "rootpath": arguments.rootpath,
                    "subgraphs": arguments.subgraphs
                })
                t.start()
                threads.append(t)
    else:
        model = arguments.model
        if arguments.model is None:
            model = "ollama_chat/llama3.3"
        if arguments.copro:
            zeroshot_copro(
                num_threads=arguments.threads,
                model=model,
                train_limit=arguments.trainlimit,
                val_limit=arguments.vallimit,
                #test_limit=arguments.testlimit,
                train_path=arguments.trainpath,
                val_path=arguments.valpath,
                #test_path=arguments.testpath,
                depth=arguments.copro_depth,
                breadth=arguments.copro_breadth,
                predictor_module=arguments.predictormodule,
                api=arguments.api,# if "ollama_chat" in arguments.model else None,
                dry_run=arguments.dry_run,
                rootpath=arguments.rootpath
            )
        elif arguments.mipro:
            zeroshot_mipro(
                model=model,
                train_limit=arguments.trainlimit,
                val_limit=arguments.vallimit,
                # test_limit=arguments.testlimit,
                train_path=arguments.trainpath,
                val_path=arguments.valpath,
                # test_path=arguments.testpath,
                num_trials=arguments.mipro_numtrials,
                minibatch_size=arguments.mipro_minibatchsize,
                minibatch_full_eval_steps=arguments.mipro_minibatchfullevalsteps,
                view_data_batch_size=arguments.mipro_viewdatabatchsize,
                predictor_module=arguments.predictormodule,
                auto=arguments.mipro_auto,
                api=arguments.api,# if "ollama_chat" in arguments.model else None,
                dry_run=arguments.dry_run,
                rootpath=arguments.rootpath
            )
        elif arguments.plain:
            zeroshot_plain(
                model=model,
                predictor_module=arguments.predictormodule,
                api=arguments.api,# if "ollama_chat" in arguments.model else None,
                dry_run=arguments.dry_run,
                rootpath=arguments.rootpath
            )
        else:
            print("Invalid Arguments")
            argparser.print_help()
    # elif arguments.evaltrain and arguments.progpath is not None:
    #     eval_program(
    #         arguments.progpath,
    #         test_path=arguments.trainpath,
    #         test_limit=arguments.trainlimit,
    #         test_result_path=arguments.outpath,
    #         num_threads=arguments.threads,
    #     )
    # elif arguments.evalval and arguments.progpath is not None:
    #     eval_program(
    #         arguments.progpath,
    #         test_path=arguments.valpath,
    #         test_limit=arguments.vallimit,
    #         test_result_path=arguments.outpath,
    #         num_threads=arguments.threads,
    #     )
    # elif arguments.evaltest and arguments.progpath is not None:
    #     eval_program(
    #         arguments.progpath,
    #         test_path=arguments.testpath,
    #         test_limit=arguments.testlimit,
    #         test_result_path=arguments.outpath,
    #         num_threads=arguments.threads,
    #     )

    for t in threads:
        t.join()
        print("Thread finished")

    print("All done")

    exit(0)