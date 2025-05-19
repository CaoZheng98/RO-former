import torch
import torch.nn as nn

import numpy as np
import random
def set_seed(seed):
    # 设置 PyTorch 的随机种子
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # 设置 NumPy 的随机种子
    np.random.seed(seed)

    # 设置 Python 的随机种子
    random.seed(seed)

    # 禁用 CUDA 的非确定性操作
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False