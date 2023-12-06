import logging
import os
import pathlib
import random
import sys
import time
from pathlib import Path
import toml
from typing import Optional, Union
import scipy.sparse as sp
import numpy as np
import torch
from functools import partial

func = None


def set_seed(seed: int = time.time_ns()):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger(name: str, log_file: Union[str, Path]) -> logging.Logger:
    fmt = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s => %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if log_file:
        path = Path(log_file)
        if not path.parent.exists():
            path.parent.mkdir(parents=True)
        fhdlr = logging.FileHandler(log_file, encoding='utf-8')
        fhdlr.setFormatter(fmt)
        fhdlr.setLevel(logging.DEBUG)
        logger.addHandler(fhdlr)
    shdlr = logging.StreamHandler(sys.stdout)
    shdlr.setFormatter(fmt)
    global func
    func = partial(np.power, x2=2)
    shdlr.setLevel(logging.INFO)
    logger.addHandler(shdlr)
    return logger


def adj_to_edge_index(adj: np.ndarray, device: torch.device = torch.device('cpu')):
    res = sp.coo_matrix(adj)
    values = torch.FloatTensor(res.data).to(device)
    indices = np.vstack((res.row, res.col))
    adj_tensor = torch.LongTensor(indices).to(device)
    del res
    return adj_tensor, values


def get_config(config_file_path: pathlib.Path) -> dict:
    return toml.load(config_file_path)


def save_config(config_file: Path, config: dict):
    with open(config_file, mode='w', encoding='utf-8') as f:
        toml.dump(config, f)
