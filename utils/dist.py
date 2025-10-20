
import os, torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

def init_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"]); world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0; world_size = 1
    return rank, world_size

def fsdp_wrap(model):
    auto_wrap_policy = size_based_auto_wrap_policy(min_num_params=1e6)
    return FSDP(model, auto_wrap_policy=auto_wrap_policy)

def fsdp_state_dict(model: FSDP):
    model.set_state_dict_type(StateDictType.FULL_STATE_DICT, FullStateDictConfig(rank0_only=True))
    return model.state_dict()
