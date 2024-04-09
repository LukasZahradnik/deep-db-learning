from argparse import ArgumentParser
from asyncio import sleep
from datetime import datetime
import math
import os, sys

from typing import List, Dict, Literal, Union, Optional, Any, Tuple, get_args


import ray


def run_cluster(cpus: int, gpus: Optional[int] = None):
    ray.init(num_cpus=cpus, num_gpus=gpus)
    sleep(60)


parser = ArgumentParser()
parser.add_argument("--cpu_devices", type=int, default=2)
parser.add_argument("--cuda_devices", type=int, default=None)

args = parser.parse_args()
print(args)

run_cluster(
    args.cpu_devices,
    args.cuda_devices,
)
