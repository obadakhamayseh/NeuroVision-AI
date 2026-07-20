

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger("brain_tumor_pipeline")

def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    logger.info(
        "Global seed fixed: seed=%d  deterministic=%s", seed, deterministic
    )
