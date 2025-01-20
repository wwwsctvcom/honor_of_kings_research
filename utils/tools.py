import random
import torch
import datetime
import numpy as np
import json
from torch.autograd import Variable
from transformers import set_seed
from loguru import logger


def get_now_time() -> str:
    """
    return: 1970-01-01_00-00-00
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_available_device():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    return device


def seed_everything(seed: int = 42) -> None:
    if seed:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def no_peak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    variable = Variable
    np_mask = variable(torch.from_numpy(np_mask) == 0)
    np_mask = np_mask.to(device)
    return np_mask


def create_masks(src, trg, device):
    src_mask = (src != -1).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != -1).unsqueeze(-2)
        trg_mask.to(device)
        size = trg.size(1)  # get seq_len for matrix
        np_mask = no_peak_mask(size, device)
        trg_mask = trg_mask & np_mask
    else:
        trg_mask = None
    return src_mask, trg_mask


def json_reader(json_name):
    try:
        with open(json_name, encoding='utf8') as f:
            ret = json.load(f)
    except Exception as e:
        logger.error(f"reading {json_name} failed, {e}")
    return ret
