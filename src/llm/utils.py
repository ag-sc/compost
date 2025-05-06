import glob
import os
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import lemon

def configure_dspy(model, basename_untimestamped, dry_run=False, api=None, rootpath=None):
    if rootpath is None:
        rootpath = os.path.join(
            os.path.dirname(sys.modules["lemon"].__file__),
            "resources"
        )

    basename = f"{basename_untimestamped}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    basepath = os.path.join(
        rootpath,
        basename
    )
    basepath_untimestamped = os.path.join(
        rootpath,
        basename_untimestamped
    )

    dirs = glob.glob(basepath_untimestamped + "*")
    if len(dirs) > 0 and any([os.path.isdir(d) for d in dirs if not d.endswith("_cache")]):# and any([d.endswith(".csv") for d in dirs]):
        print("File already exists, skipping", dirs)
        raise ValueError("File already exists, skipping " + str(dirs))
    else:
        print("File does not exist or not finished", basepath_untimestamped + "*", flush=True)

    if dry_run:
        raise ValueError("Dry run")

    sys.stdout = open(Path(basepath).with_suffix('.log'), "a+")
    sys.stderr = open(Path(basepath).with_suffix('.err'), "a+")

    cache_path = Path(os.path.join(
        rootpath,
        basename_untimestamped + "_cache"
    ))
    cache_path.mkdir(parents=True, exist_ok=True)

    os.environ["DSPY_CACHEDIR"] = str(cache_path)
    os.environ["DSP_CACHEDIR"] = str(cache_path)

    max_tokens = 2*4096
    if "olmo" in model:
        max_tokens = 4096

    import dspy  # type: ignore

    print("Configuring dspy", model, api, os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '', flush=True)

    if api is not None: # and not model.startswith("openai"):
        lm = dspy.LM(model,
                     api_base=api,
                     api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
                     drop_params=True,
                     cache=True,
                     timeout=60*60*24,
                     max_tokens=max_tokens)
    else:
        lm = dspy.LM(model,
                     api_key=os.environ["OPENAI_API_KEY"] if model.startswith("openai") else '',
                     drop_params=False if model.startswith("openai") else True,
                     cache=True,
                     timeout=60*60*24,
                     max_tokens=max_tokens)
    dspy.configure(lm=lm, timeout=60*60*24, max_tokens=max_tokens)

    dspy.enable_logging()
    dspy.enable_litellm_logging()

    print(lm("Say this is a test!", temperature=0.7), flush=True)

    return basepath


def run_nvidia_smi():
    hostname = socket.gethostname()
    pid = os.getpid()
    filename = f"{hostname}_{pid}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.txt"

    with open(filename, "a+") as file:
        while True:
            try:
                # Run the nvidia-smi command
                result = subprocess.run(['/usr/bin/nvidia-smi'], capture_output=True, text=True, check=True)
                # Print the output
                file.write(result.stdout + "\n")
                file.flush()
            except subprocess.CalledProcessError as e:
                print(f"Error running nvidia-smi: {e}")
            except FileNotFoundError:
                print("nvidia-smi command not found. Ensure NVIDIA drivers are installed.")
            # Wait for 30 seconds before the next execution
            time.sleep(30)
