import logging
import functools

from torch.distributed import get_rank, get_world_size, new_group, barrier

logger = logging.getLogger(__name__)


class DistributedManager():
    distributed = False
    grp = None
    local_rank = 0

    @staticmethod
    def set_args(args):
        DistributedManager.distributed = args.distributed
        if args.distributed:
            logger.info("[{}] >> Setting barrier for group: {} ".format(get_rank(), list(range(0, get_world_size()))))
            DistributedManager.grp = new_group(list(range(0, get_world_size())))
            DistributedManager.local_rank = args.local_rank

    @staticmethod
    def is_master():
        return (not DistributedManager.distributed) or DistributedManager.local_rank == 0

    @staticmethod
    def get_rank_():
        rank = 0 if not DistributedManager.distributed else get_rank()
        return rank

    @staticmethod
    def is_first():
        return DistributedManager.get_rank_() == 0

    @staticmethod
    def set_barrier():
        if DistributedManager.distributed:
            logger.info("[{}] >> Barrier waiting ".format(get_rank()))
            barrier(group=DistributedManager.grp)
            logger.info("[{}] >> Barrier passed ".format(get_rank()))


# Decorator that forces the decorated function to run only on master node
# When not running in distributed - the function will always run
def master_only(func):

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if DistributedManager.is_master():
            return func(*args, **kwargs)

    return wrapper

@master_only
def print_at_master(str):
    print(str)