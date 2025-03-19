import random

import numpy as np

try:
    import torch
except ImportError:
    torch = None


def seed_everything(seed: int) -> None:
    r"""Sets the seed for generating random number in Pytorch, numpy and Python.

    Args:
        seed(int): The desired seed.

    """
    random.seed(seed)
    np.random.seed(seed)

    if torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
