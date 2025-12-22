import os
import sys
import torch
import logging
import random
import argparse
import traceback
import torch.distributed as dist

from hydra import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import OmegaConf
from training.sam2.training.utils.train_utils import makedir, register_omegaconf_resolvers


def format_exception(e: Exception, limit=20):
    traceback_str = "".join(traceback.format_tb(e.__traceback__, limit=limit))
    return f"{type(e).__name__}: {e}\nTraceback:\n{traceback_str}"


def add_pythonpath_to_sys_path():
    if "PYTHONPATH" in os.environ and os.environ["PYTHONPATH"]:
        sys.path = os.environ["PYTHONPATH"].split(":") + sys.path


def setup_env():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]

    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl",
        init_method=f"env://",
        world_size=world_size,
        rank=rank
    )
    return rank, local_rank, world_size


if __name__ == "__main__":
    initialize_config_module("sam2", version_base="1.2")

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="Path to config file")
    args = parser.parse_args()

    cfg = compose(config_name=args.config)
    add_pythonpath_to_sys_path()
    register_omegaconf_resolvers()

    if cfg.launcher.experiment_log_dir is None:
        cfg.launcher.experiment_log_dir = os.path.join(os.getcwd(), "logs", args.config)
    makedir(cfg.launcher.experiment_log_dir)

    with open(os.path.join(cfg.launcher.experiment_log_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    rank, local_rank, world_size = setup_env()

    try:
        cfg_resolved = OmegaConf.to_container(cfg, resolve=False)
        cfg_resolved = OmegaConf.create(cfg_resolved)

        trainer = instantiate(cfg_resolved.trainer, _recursive_=False)
        trainer.run()
    except Exception as e:
        message = format_exception(e)
        logging.error(message)
        raise e