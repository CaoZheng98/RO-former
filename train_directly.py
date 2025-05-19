from data.dataset import DT_Dataset
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from utils.collate_fn import collate_fn
import torch
from utils.set_seed import set_seed
from torch import nn
from utils.metrics import loss_func_mse
from utils.metrics import cal_nse_torch
# from pretrain import saving_root
from utils.lr_strategies import SchedulerFactory
DS = DT_Dataset

def __main__():
    from models.transformer.transformer import Transformer
    from utils.train_full import train_full
    # 设置随机种子
    seed = 42
    set_seed(seed)
    max_len = 2000
    pad_idx = 1
    bos_len = 3
    learning_rate = 1e-2
    n_epochs = 500
    d_model = 64
    batch_size = 100
    # _________dataset________
    out_fall_name = 'JHHHHC'
    Rainfall_intensity = 3
    max_patience = 12
    ds_train = DS(outfall_name = out_fall_name,  Rainfall_intensity = Rainfall_intensity,max_patience=max_patience)
    ds_val = DS(outfall_name=out_fall_name,  Rainfall_intensity=Rainfall_intensity, max_patience=max_patience,type='val')
    train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(dataset=ds_val, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(src_len_max = max_len,trg_len_max=max_len, src_pad_idx = pad_idx, trg_pad_idx=pad_idx, d_src=1, d_trg=1, d_model=d_model, trg_size=1).to(device)
    # 如果有多个 GPU，使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": n_epochs * 0.25, "decay_rate":0.99}
    scheduler = SchedulerFactory.get_scheduler(optimizer, **scheduler_paras)
    saving_root = Path("./runs") / f'ml_{max_len}_padidx_{pad_idx}_bos_len_{bos_len}_lr_{learning_rate}_epochs_{n_epochs}_d_model_{d_model}_of_{out_fall_name}_RI_{Rainfall_intensity}'
    train_full(model, train_loader, val_loader, optimizer,bos_len,scheduler, loss_func_mse, n_epochs, device, saving_root)
__main__()