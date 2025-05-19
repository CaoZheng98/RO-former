
import torch
from models.transformer.transformer import Transformer  # 替换为你的模型模块路径
from pathlib import Path
from utils.train_full import train_full
from utils import set_seed
from data.dataset_ import DT_Dataset
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from utils.collate_fn import collate_fn
import torch
from utils.set_seed import set_seed
from torch import nn
from utils.metrics import loss_func
from utils.metrics import cal_nse_torch
# from pretrain import saving_root
from utils.lr_strategies import SchedulerFactory
DS = DT_Dataset
def load_model(model_path, device):
    """
    加载保存的模型状态字典并应用到模型实例中。
    :param model_path: 保存的模型文件路径
    :param device: 设备 (CPU 或 GPU)
    :return: 加载状态字典后的模型实例
    """
    # 创建模型实例
    max_len = 2000  # 替换为实际的 max_len
    pad_idx = 1     # 替换为实际的 pad_idx
    d_model = 64    # 替换为实际的 d_model
    model = Transformer(src_len_max=max_len, trg_len_max=max_len, src_pad_idx=pad_idx, trg_pad_idx=pad_idx, d_src=1, d_trg=1, d_model=d_model, trg_size=1).to(device)
    # 加载保存的模型状态字典
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 指定保存的模型文件路径
saving_root = Path("./runs") / f'ml_2000_padidx_1_bos_len_3_lr_0.01_epochs_200_d_model_64_of_JHHXY_RI_10'
model_path = saving_root / "(newest)_199_199.pkl"
# 加载模型
model = load_model(model_path, device)
# 设置随机种子
d_model = 64
seed = 42
set_seed(seed)
max_len = 2000
pad_idx = 1
bos_len = 3
learning_rate = 1e-2
n_epochs = 200
batch_size = 100
out_fall_name = 'JHHXY'
Rainfall_intensity = 10
ds_train = DS(outfall_name=out_fall_name, Rainfall_intensity=Rainfall_intensity)
train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 如果有多个 GPU，使用 DataParallel 包装模型
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": n_epochs * 0.25, "decay_rate": 1}
scheduler = SchedulerFactory.get_scheduler(optimizer, **scheduler_paras)
saving_root = Path(
    "./runs") / f'fine_tune_ml_{max_len}_padidx_{pad_idx}_bos_len_{bos_len}_lr_{learning_rate}_epochs_{n_epochs}_d_model_{d_model}_of_{out_fall_name}_RI_{Rainfall_intensity}'

train_full(model, train_loader, val_loader, optimizer,bos_len,scheduler, loss_func, n_epochs, device, saving_root)