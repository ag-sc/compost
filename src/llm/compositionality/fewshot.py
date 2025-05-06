import argparse
import csv
import io
#import datetime
import os
import threading
from datetime import datetime
from multiprocessing import Process
from pathlib import Path
from typing import Literal

from llm.compositionality import evaluation


# cache_path = Path(f"/dev/shm/{os.getpid()}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
# cache_path.mkdir(parents=True, exist_ok=True)
#
# os.environ["DSPY_CACHEDIR"] = str(cache_path)
# os.environ["DSP_CACHEDIR"] = str(cache_path)
# import dspy  # type: ignore
import more_itertools

#from dspy import MIPROv2, LabeledFewShot, BootstrapFewShot, BootstrapFewShotWithRandomSearch

from dudes import consts
from dudes.qa.sparql.sparql_endpoint import SPARQLEndpoint
from dudes.utils import slugify
from llm.compositionality.dataset import load_datasets
from llm.compositionality.evaluation import validate_context_and_answer
from llm.utils import configure_dspy, run_nvidia_smi


def fewshot_plain(
    model = "ollama_chat/llama3.3",
    predictor_module = None,
    fixed = True,
    api=None,
    dry_run=False,
    rootpath=None,
    subgraphs=False,
):
    basename_untimestamped = f"fewshot_plain_{'fixed' if fixed else 'nonfix'}_{slugify(predictor_module)}_{slugify(model.split('/')[-1])}"
    try:
        basepath = configure_dspy(model, basename_untimestamped, dry_run=dry_run, api=api, rootpath=rootpath)
    except ValueError as e:
        print(e)
        return

    if fixed:
        if subgraphs:
            from llm.compositionality.prompting_modules import FewShotVariableFixedPipeline
            baseline = FewShotVariableFixedPipeline(predictor_module=predictor_module)
        else:
            from llm.compositionality.prompting_modules import FewShotFixedPipeline
            baseline = FewShotFixedPipeline(predictor_module=predictor_module)
    else:
        from llm.compositionality.prompting_modules import FewShotPipeline
        baseline = FewShotPipeline(predictor_module=predictor_module)
    print(baseline)

    baseline.save(basepath, save_program=True)
    baseline.save(str(Path(basepath).with_suffix('.json')), save_program=False)
    print("Saved to", basepath)

def fewshot_copro(
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
        #fixed=True,
        api=None,
        dry_run=False,
        rootpath=None,
        subgraphs=False,
):
    basename_untimestamped = f"fewshot_copro_fixed_{slugify(predictor_module)}_{slugify(model.split('/')[-1])}_{train_limit}_{val_limit}_{depth}_{breadth}"
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
        fixed_fewshot=True,
        subgraphs=subgraphs,
    )

    if subgraphs:
        from llm.compositionality.prompting_modules import FewShotVariableFixedPipeline
        baseline = FewShotVariableFixedPipeline(predictor_module=predictor_module)
    else:
        from llm.compositionality.prompting_modules import FewShotFixedPipeline
        baseline = FewShotFixedPipeline(predictor_module=predictor_module)
    print(baseline)

    import dspy
    teleprompter = dspy.COPRO(
        metric=validate_context_and_answer,
        verbose=True,
        breadth=breadth,
        depth=depth
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

def fewshot_mipro(
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
        fixed=False,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        auto: Literal["light", "medium", "heavy"]="light",
        api=None,
        dry_run=False,
        rootpath=None,
        subgraphs=False,
):
    if fixed and (max_bootstrapped_demos > 0 or max_labeled_demos > 0):
        raise ValueError("Cannot use fixed=True with max_bootstrapped_demos or max_labeled_demos")

    basename_untimestamped = f"fewshot_mipro_{'fixed' if fixed else 'nonfix'}_{auto}_{slugify(predictor_module)}_{slugify(model.split('/')[-1])}_{train_limit}_{val_limit}_{num_trials}_{minibatch_size}_{minibatch_full_eval_steps}_{view_data_batch_size}_{max_bootstrapped_demos}_{max_labeled_demos}"
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
        fixed_fewshot=fixed,
        subgraphs=subgraphs,
    )

    if fixed:
        if subgraphs:
            from llm.compositionality.prompting_modules import FewShotVariableFixedPipeline
            baseline = FewShotVariableFixedPipeline(predictor_module=predictor_module)
        else:
            from llm.compositionality.prompting_modules import FewShotFixedPipeline
            baseline = FewShotFixedPipeline(predictor_module=predictor_module)
    else:
        from llm.compositionality.prompting_modules import FewShotPipeline
        baseline = FewShotPipeline(predictor_module=predictor_module)
    print(baseline)

    # Initialize optimizer
    from dspy import MIPROv2
    teleprompter = MIPROv2(
        metric=validate_context_and_answer,
        auto=auto,  # Can choose between light, medium, and heavy optimization runs
        max_errors=2*len(train), #ignore errors
        verbose=True
    )

    # Optimize program
    print(f"Optimizing program with MIPRO...")
    optimized_program = teleprompter.compile(
        baseline.deepcopy(),
        trainset=train,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
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

def fewshot_labeledfewshot(
        train_path=None,
        val_path=None,
        model="ollama_chat/llama3.3",
        train_limit=None,
        val_limit=None,
        predictor_module=None,
        k=16,
        api=None,
        dry_run=False,
        rootpath=None,
        subgraphs=False,
):
    basename_untimestamped = f"fewshot_labeledfewshot_nonfix_{slugify(predictor_module)}_{slugify(model.split('/')[-1])}_{train_limit}_{val_limit}_{k}"
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
        subgraphs=subgraphs,
    )

    from llm.compositionality.prompting_modules import FewShotPipeline
    baseline = FewShotPipeline(predictor_module=predictor_module)
    print(baseline)

    # Initialize optimizer
    from dspy import LabeledFewShot
    teleprompter = LabeledFewShot(
        k=k,
    )

    optimized_program = teleprompter.compile(
        baseline.deepcopy(),
        trainset=train,
        sample=True,
    )

    print(optimized_program)

    optimized_program.save(basepath, save_program=True)
    optimized_program.save(str(Path(basepath).with_suffix('.json')), save_program=False)
    print("Saved to", basepath)

def fewshot_bootstrapfewshot(
        train_path=None,
        val_path=None,
        model="ollama_chat/llama3.3",
        train_limit=None,
        val_limit=None,
        predictor_module=None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=5,
        api=None,
        dry_run=False,
        rootpath=None,
        subgraphs=False,
):
    basename_untimestamped = f"fewshot_bootstrapfewshot_nonfix_{slugify(predictor_module)}_{slugify(model.split('/')[-1])}_{train_limit}_{val_limit}_{max_bootstrapped_demos}_{max_labeled_demos}_{max_rounds}"
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
        subgraphs=subgraphs,
    )

    from llm.compositionality.prompting_modules import FewShotPipeline
    baseline = FewShotPipeline(predictor_module=predictor_module)
    print(baseline)

    # Initialize optimizer
    from dspy import BootstrapFewShot
    teleprompter = BootstrapFewShot(
        metric=validate_context_and_answer,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
        max_errors=2 * len(train),  # ignore errors
    )

    optimized_program = teleprompter.compile(
        baseline.deepcopy(),
        trainset=train,
    )

    print(optimized_program)

    optimized_program.save(basepath, save_program=True)
    optimized_program.save(str(Path(basepath).with_suffix('.json')), save_program=False)
    print("Saved to", basepath)

def fewshot_bootstrapfewshotrandomsearch(
        train_path=None,
        val_path=None,
        model="ollama_chat/llama3.3",
        train_limit=None,
        val_limit=None,
        predictor_module=None,
        max_bootstrapped_demos=4,
        max_labeled_demos=16,
        max_rounds=5,
        num_candidate_programs=16,
        num_threads=8,
        api=None,
        dry_run=False,
        rootpath=None,
        subgraphs=False,
):
    basename_untimestamped = f"fewshot_bootstrapfewshotrandomsearch_nonfix_{slugify(predictor_module)}_{slugify(model.split('/')[-1])}_{train_limit}_{val_limit}_{max_bootstrapped_demos}_{max_labeled_demos}_{max_rounds}_{num_candidate_programs}_{num_threads}"
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
        subgraphs=subgraphs,
    )

    from llm.compositionality.prompting_modules import FewShotPipeline
    baseline = FewShotPipeline(predictor_module=predictor_module)
    print(baseline)

    # Initialize optimizer
    from dspy import BootstrapFewShotWithRandomSearch
    teleprompter = BootstrapFewShotWithRandomSearch(
        metric=validate_context_and_answer,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        max_rounds=max_rounds,
        num_candidate_programs=num_candidate_programs,
        num_threads=num_threads,
        max_errors=2 * len(train) + 2 * len(val),  # ignore errors
    )

    optimized_program = teleprompter.compile(
        baseline.deepcopy(),
        trainset=train,
        valset=val,
    )

    print(optimized_program)

    optimized_program.save(basepath, save_program=True)
    optimized_program.save(str(Path(basepath).with_suffix('.json')), save_program=False)
    print("Saved to", basepath)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model", type=str, default=None)
    argparser.add_argument("--threads", type=int, default=8)
    argparser.add_argument("--trainlimit", type=int, default=None)
    argparser.add_argument("--vallimit", type=int, default=None)
    argparser.add_argument("--trainpath", type=str, default=None)
    argparser.add_argument("--valpath", type=str, default=None)
    argparser.add_argument("--predictormodule", type=str, default=None)
    argparser.add_argument('--fixed', action=argparse.BooleanOptionalAction)
    argparser.add_argument('--plain', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--api", type=str, default=None)
    argparser.add_argument("--endpoint", type=str, default=None)

    argparser.add_argument('--copro', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--copro-depth", type=int, default=3)
    argparser.add_argument("--copro-breadth", type=int, default=10)

    argparser.add_argument('--mipro', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--mipro-numtrials", type=int, default=30)
    argparser.add_argument("--mipro-minibatchsize", type=int, default=25)
    argparser.add_argument("--mipro-minibatchfullevalsteps", type=int, default=10)
    argparser.add_argument("--mipro-viewdatabatchsize", type=int, default=10)
    argparser.add_argument("--mipro-auto", type=str, default="light", choices=["light", "medium", "heavy"])
    argparser.add_argument("--mipro-maxbootstrappeddemos", type=int, default=4)
    argparser.add_argument("--mipro-maxlabeleddemos", type=int, default=16)

    argparser.add_argument('--labeledfewshot', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--labeledfewshot-k", type=int, default=16)

    argparser.add_argument('--bootstrapfewshot', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--bootstrapfewshot-maxbootstrappeddemos", type=int, default=4)
    argparser.add_argument("--bootstrapfewshot-maxlabeleddemos", type=int, default=16)
    argparser.add_argument("--bootstrapfewshot-maxrounds", type=int, default=5)

    argparser.add_argument('--bootstrapfewshotrandomsearch', action=argparse.BooleanOptionalAction)
    argparser.add_argument("--bootstrapfewshotrandomsearch-maxbootstrappeddemos", type=int, default=4)
    argparser.add_argument("--bootstrapfewshotrandomsearch-maxlabeleddemos", type=int, default=16)
    argparser.add_argument("--bootstrapfewshotrandomsearch-maxrounds", type=int, default=5)
    argparser.add_argument("--bootstrapfewshotrandomsearch-numcandidateprograms", type=int, default=16)

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

    # fewshot_mipro(
    #     model="ollama_chat/phi4",
    #     train_limit=100,
    #     val_limit=100,
    #     max_bootstrapped_demos=0,
    #     max_labeled_demos=0,
    #     train_path="./resources/compositionality_subgraph_hard_prompt_5.json.zst",
    #     predictor_module=None,
    #     auto="light",
    #     fixed=True,
    #     rootpath="./",
    #     subgraphs=True
    # )


    if arguments.endpoint is not None:
        consts.endpoint = arguments.endpoint
        evaluation.sparql_endpoint = SPARQLEndpoint(endpoint=arguments.endpoint, cache=dict(), check_endpoint=False, default_graph="http://dbpedia.org")

    #assert arguments.batch_frac > 0.0 and arguments.batch_frac <= 1.0
    #assert arguments.batch_id >= 0

    threads = []

    if arguments.autoexp or arguments.autoexpcsv is not None:
        if arguments.autoexp:
            experiments = list(set([
                (
                    func,
                    model,
                    predictor_module,
                    auto if func == "mipro" else None,
                    fixed if func in ["plain", "mipro"] else None
                )
                for model in (["ollama_chat/llama3.3", "ollama_chat/phi4", "ollama_chat/olmo2:7b-1124-instruct-q4_K_M", "ollama_chat/qwen2.5-coder", "openai/gpt-4o-mini-2024-07-18"] if arguments.model is None else [arguments.model])#, "openai/gpt-4o-mini", "openai/deepseek-r1"
                for func in ["mipro", "labeledfewshot", "bootstrapfewshot", "bootstrapfewshotrandomsearch"] # "plain", "copro",
                for predictor_module in ["chainofthought", None]
                for auto in ["light", "medium", "heavy"]
                for fixed in ["True", "False"]
                # broader exclusion to avoid GPT + Copro as it requires fixed shots
                # (not just deepseek which appears to be incompatible with copro when using llama.cpp)
            ]))
            experiments.sort(key=lambda x: str(x[1:2]+x[0:1]+x[2:]))

            output = io.StringIO()
            csv_writer = csv.writer(output)
            csv_writer.writerows(experiments)
            csv_string = output.getvalue()
            print(csv_string)
            output.close()
            exit(0)
            #print("Total Experiments", experiments)

        # plain_exps = [d for d in experiments if d[0] == "plain"]
        # nonplain_exps = [d for d in experiments if d[0] != "plain"]
        # nonplain_exps.sort(key=lambda x: str(x))
        #
        # experiments = list(more_itertools.chunked(nonplain_exps, int(len(nonplain_exps) * arguments.batch_frac) + 1))[arguments.batch_id]
        #
        # if arguments.batch_id == 0:
        #     experiments += plain_exps
        #
        # print("Running Experiments", experiments)

        if arguments.autoexpcsv is not None:
            with open(arguments.autoexpcsv) as f:#tuple(line) for idx, line in enumerate(csv.reader(f)) if arguments.autoexpcsvid is None or idx == arguments.autoexpcsvid
                experiments = [tuple([True if l == "True" else (False if l == "False" else l) for l in line]) for idx, line in enumerate(csv.reader(f)) if arguments.autoexpcsvid is None or idx == arguments.autoexpcsvid]

        thread = threading.Thread(target=run_nvidia_smi, daemon=True)
        thread.start()

        for func, model, predictor_module, auto, fixed in experiments:
            if func == "plain":
                t = Process(target=fewshot_plain, kwargs=dict(
                    model=model,
                    predictor_module=predictor_module,
                    fixed=fixed,
                    api=arguments.api,
                    dry_run=arguments.dry_run,
                    rootpath=arguments.rootpath,
                    subgraphs=arguments.subgraphs
                ))
                t.start()
                threads.append(t)
                # fewshot_plain(
                #     model=arguments.model,
                #     predictor_module=arguments.predictormodule,
                #     fixed=arguments.fixed,
                #     api=arguments.api
                # )
            elif func == "copro":
                t = Process(target=fewshot_copro, kwargs=dict(
                    num_threads=arguments.threads,
                    model=model,
                    train_limit=arguments.trainlimit,
                    val_limit=arguments.vallimit,
                    train_path=arguments.trainpath,
                    val_path=arguments.valpath,
                    depth=arguments.copro_depth,
                    breadth=arguments.copro_breadth,
                    predictor_module=predictor_module,
                    api=arguments.api,
                    dry_run=arguments.dry_run,
                    rootpath=arguments.rootpath,
                    subgraphs=arguments.subgraphs
                ))
                t.start()
                threads.append(t)
                # fewshot_copro(
                #     num_threads=arguments.threads,
                #     model=arguments.model,
                #     train_limit=arguments.trainlimit,
                #     val_limit=arguments.vallimit,
                #     # test_limit=arguments.testlimit,
                #     train_path=arguments.trainpath,
                #     val_path=arguments.valpath,
                #     # test_path=arguments.testpath,
                #     depth=arguments.copro_depth,
                #     breadth=arguments.copro_breadth,
                #     predictor_module=arguments.predictormodule,
                #     api=arguments.api
                # )
            elif func == "mipro":
                t = Process(target=fewshot_mipro, kwargs=dict(
                    model=model,
                    train_limit=arguments.trainlimit,
                    val_limit=arguments.vallimit,
                    train_path=arguments.trainpath,
                    val_path=arguments.valpath,
                    num_trials=arguments.mipro_numtrials,
                    minibatch_size=arguments.mipro_minibatchsize,
                    minibatch_full_eval_steps=arguments.mipro_minibatchfullevalsteps,
                    view_data_batch_size=arguments.mipro_viewdatabatchsize,
                    predictor_module=predictor_module,
                    auto=auto,
                    max_bootstrapped_demos=arguments.mipro_maxbootstrappeddemos if not fixed else 0,
                    max_labeled_demos=arguments.mipro_maxlabeleddemos if not fixed else 0,
                    fixed=fixed,
                    api=arguments.api,
                    dry_run=arguments.dry_run,
                    rootpath=arguments.rootpath,
                    subgraphs=arguments.subgraphs
                ))
                t.start()
                threads.append(t)
                # fewshot_mipro(
                #     model=arguments.model,
                #     train_limit=arguments.trainlimit,
                #     val_limit=arguments.vallimit,
                #     # test_limit=arguments.testlimit,
                #     train_path=arguments.trainpath,
                #     val_path=arguments.valpath,
                #     # test_path=arguments.testpath,
                #     num_trials=arguments.mipro_numtrials,
                #     minibatch_size=arguments.mipro_minibatchsize,
                #     minibatch_full_eval_steps=arguments.mipro_minibatchfullevalsteps,
                #     view_data_batch_size=arguments.mipro_viewdatabatchsize,
                #     predictor_module=arguments.predictormodule,
                #     auto=arguments.mipro_auto,
                #     max_bootstrapped_demos=arguments.mipro_maxbootstrappeddemos if not arguments.fixed else 0,
                #     max_labeled_demos=arguments.mipro_maxlabeleddemos if not arguments.fixed else 0,
                #     fixed=arguments.fixed,
                #     api=arguments.api
                # )
            elif func == "labeledfewshot":
                t = Process(target=fewshot_labeledfewshot, kwargs=dict(
                    model=model,
                    train_limit=arguments.trainlimit,
                    val_limit=arguments.vallimit,
                    train_path=arguments.trainpath,
                    val_path=arguments.valpath,
                    predictor_module=predictor_module,
                    k=arguments.labeledfewshot_k,
                    api=arguments.api,
                    dry_run=arguments.dry_run,
                    rootpath=arguments.rootpath,
                    subgraphs=arguments.subgraphs
                ))
                t.start()
                threads.append(t)
                # fewshot_labeledfewshot(
                #     model=arguments.model,
                #     train_limit=arguments.trainlimit,
                #     val_limit=arguments.vallimit,
                #     train_path=arguments.trainpath,
                #     val_path=arguments.valpath,
                #     predictor_module=arguments.predictormodule,
                #     k=arguments.labeledfewshot_k,
                #     api=arguments.api
                # )
            elif func == "bootstrapfewshot":
                t = Process(target=fewshot_bootstrapfewshot, kwargs=dict(
                    model=model,
                    train_limit=arguments.trainlimit,
                    val_limit=arguments.vallimit,
                    train_path=arguments.trainpath,
                    val_path=arguments.valpath,
                    predictor_module=predictor_module,
                    max_bootstrapped_demos=arguments.bootstrapfewshot_maxbootstrappeddemos,
                    max_labeled_demos=arguments.bootstrapfewshot_maxlabeleddemos,
                    max_rounds=arguments.bootstrapfewshot_maxrounds,
                    api=arguments.api,
                    dry_run=arguments.dry_run,
                    rootpath=arguments.rootpath,
                    subgraphs=arguments.subgraphs
                ))
                t.start()
                threads.append(t)
                # fewshot_bootstrapfewshot(
                #     model=arguments.model,
                #     train_limit=arguments.trainlimit,
                #     val_limit=arguments.vallimit,
                #     train_path=arguments.trainpath,
                #     val_path=arguments.valpath,
                #     predictor_module=arguments.predictormodule,
                #     max_bootstrapped_demos=arguments.bootstrapfewshot_maxbootstrappeddemos,
                #     max_labeled_demos=arguments.bootstrapfewshot_maxlabeleddemos,
                #     max_rounds=arguments.bootstrapfewshot_maxrounds,
                #     api=arguments.api
                # )
            elif func == "bootstrapfewshotrandomsearch":
                t = Process(target=fewshot_bootstrapfewshotrandomsearch, kwargs=dict(
                    model=model,
                    train_limit=arguments.trainlimit,
                    val_limit=arguments.vallimit,
                    train_path=arguments.trainpath,
                    val_path=arguments.valpath,
                    predictor_module=predictor_module,
                    max_bootstrapped_demos=arguments.bootstrapfewshot_maxbootstrappeddemos,
                    max_labeled_demos=arguments.bootstrapfewshot_maxlabeleddemos,
                    max_rounds=arguments.bootstrapfewshot_maxrounds,
                    num_candidate_programs=arguments.bootstrapfewshotrandomsearch_numcandidateprograms,
                    num_threads=arguments.threads,
                    api=arguments.api,
                    dry_run=arguments.dry_run,
                    rootpath=arguments.rootpath,
                    subgraphs=arguments.subgraphs
                ))
                t.start()
                threads.append(t)
                # fewshot_bootstrapfewshotrandomsearch(
                #     model=arguments.model,
                #     train_limit=arguments.trainlimit,
                #     val_limit=arguments.vallimit,
                #     train_path=arguments.trainpath,
                #     val_path=arguments.valpath,
                #     predictor_module=arguments.predictormodule,
                #     max_bootstrapped_demos=arguments.bootstrapfewshot_maxbootstrappeddemos,
                #     max_labeled_demos=arguments.bootstrapfewshot_maxlabeleddemos,
                #     max_rounds=arguments.bootstrapfewshot_maxrounds,
                #     num_candidate_programs=arguments.bootstrapfewshotrandomsearch_numcandidateprograms,
                #     num_threads=arguments.threads,
                #     api=arguments.api
                # )
    else:
        model = arguments.model
        if model is None:
            model = "ollama_chat/llama3.3"

        if arguments.copro:
            fewshot_copro(
                num_threads=arguments.threads,
                model=model,
                train_limit=arguments.trainlimit,
                val_limit=arguments.vallimit,
                # test_limit=arguments.testlimit,
                train_path=arguments.trainpath,
                val_path=arguments.valpath,
                # test_path=arguments.testpath,
                depth=arguments.copro_depth,
                breadth=arguments.copro_breadth,
                predictor_module=arguments.predictormodule,
                api=arguments.api,
                rootpath=arguments.rootpath,
                subgraphs=arguments.subgraphs
            )
        elif arguments.mipro:
            fewshot_mipro(
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
                max_bootstrapped_demos=arguments.mipro_maxbootstrappeddemos if not arguments.fixed else 0,
                max_labeled_demos=arguments.mipro_maxlabeleddemos if not arguments.fixed else 0,
                fixed=arguments.fixed,
                api=arguments.api,
                rootpath=arguments.rootpath,
                subgraphs=arguments.subgraphs
            )
        elif arguments.plain:
            fewshot_plain(
                model=model,
                predictor_module=arguments.predictormodule,
                fixed=arguments.fixed,
                api=arguments.api,
                rootpath=arguments.rootpath,
                subgraphs=arguments.subgraphs
            )
        elif arguments.labeledfewshot:
            fewshot_labeledfewshot(
                model=model,
                train_limit=arguments.trainlimit,
                val_limit=arguments.vallimit,
                train_path=arguments.trainpath,
                val_path=arguments.valpath,
                predictor_module=arguments.predictormodule,
                k=arguments.labeledfewshot_k,
                api=arguments.api,
                rootpath=arguments.rootpath,
                subgraphs=arguments.subgraphs
            )
        elif arguments.bootstrapfewshot:
            fewshot_bootstrapfewshot(
                model=model,
                train_limit=arguments.trainlimit,
                val_limit=arguments.vallimit,
                train_path=arguments.trainpath,
                val_path=arguments.valpath,
                predictor_module=arguments.predictormodule,
                max_bootstrapped_demos=arguments.bootstrapfewshot_maxbootstrappeddemos,
                max_labeled_demos=arguments.bootstrapfewshot_maxlabeleddemos,
                max_rounds=arguments.bootstrapfewshot_maxrounds,
                api=arguments.api,
                rootpath=arguments.rootpath,
                subgraphs=arguments.subgraphs
            )
        elif arguments.bootstrapfewshotrandomsearch:
            fewshot_bootstrapfewshotrandomsearch(
                model=model,
                train_limit=arguments.trainlimit,
                val_limit=arguments.vallimit,
                train_path=arguments.trainpath,
                val_path=arguments.valpath,
                predictor_module=arguments.predictormodule,
                max_bootstrapped_demos=arguments.bootstrapfewshot_maxbootstrappeddemos,
                max_labeled_demos=arguments.bootstrapfewshot_maxlabeleddemos,
                max_rounds=arguments.bootstrapfewshot_maxrounds,
                num_candidate_programs=arguments.bootstrapfewshotrandomsearch_numcandidateprograms,
                num_threads=arguments.threads,
                api=arguments.api,
                rootpath=arguments.rootpath,
                subgraphs=arguments.subgraphs
            )
        else:
            print("Invalid Arguments")
            argparser.print_help()

    for t in threads:
        t.join()
        print("Thread finished")

    print("All done")

    exit(0)