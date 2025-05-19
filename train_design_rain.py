from data.dataset import design_rain_Dataset, real_Dataset
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn
from utils.lr_strategies import SchedulerFactory
# from pretrain import savingoot
DS = design_rain_Dataset
RD =real_Dataset
out_fall_path = r"./data/swmm/output1.pkl"
rain_off_path = r"./data/swmm/rain_off.pkl"
val_path = r"./data/swmm/real_ROdata_2023_4_3.pkl"
saving_root = Path("./runs")/'design_rain'
ds_train = DS(out_fall_path, rain_off_path)
ds_val = RD(val_path)
batch_size = 32


def collate_fn(batch):
    # 假设每个样本是一个元组 (features, labels)，其中 features 是 numpy.ndarray 类型
    features, labels = zip(*batch)

    for label in labels:
        while label and label[-1] == 0:
            label.pop()

    # 将 numpy.ndarray 转换为 Tensor
    features = [torch.tensor(feature) for feature in features]
    labels = [torch.tensor(label) for label in labels]  # 假设标签已经是 Tensor 类型
    # print("len_features:", len(features[1]))
    # print("labels:", len(labels))
    max_length1 = max(len(feature) for feature in features)
    max_length2 = max(len(label) for label in labels)

    # 对每个样本的特征进行填充
    padded_features = [torch.cat([feature, torch.ones(max_length1 - len(feature))]) for feature in
                       features]
    padded_labels = [torch.cat([label, torch.ones(max_length2 - len(label))]) for label in labels]

    padded_labels_ = [torch.cat([label, torch.zeros(max_length2 - len(label))]) for label in labels]
    # # 将填充后的特征和标签重新组合成元组
    # padded_batch = list(zip(padded_features, labels))
    # 将填充后的特征堆叠成一个张量，标签保持不变
    return torch.stack(padded_features), torch.stack(padded_labels), torch.stack(padded_labels_)

def loss_func(y_hat, trg, bos_len=3,pad_idx=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cal_loss = nn.MSELoss().to(device)
    mask = (trg != pad_idx)
    y_hat = y_hat.squeeze(-1)
    pred = y_hat.masked_fill(mask == False, 0)[:][bos_len-1:-1] if (y_hat.shape[0]>1) else y_hat.masked_fill(mask == False, 0)[0][bos_len-1:-1]
    ground_truth = trg.masked_fill(mask == False, 0)[:][bos_len:] if (trg.shape[0]>1) else trg.masked_fill(mask == False, 0)[0][bos_len:]

    loss = cal_loss(pred, ground_truth)
    return loss
def __main__():
    from models.transformer.transformer import Transformer
    from utils.train_full import train_full
    train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(dataset=ds_val, batch_size=1, shuffle=False,collate_fn=collate_fn)   #待修改
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(src_len_max=2000, trg_len_max=2000, src_pad_idx=1, trg_pad_idx=1, d_src=1, d_trg=1, d_model=64,
                        trg_size=1).to(device)
    bos_len = 1
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    n_epochs = 60
    scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": n_epochs * 0.25, "decay_rate": 0.99}
    scheduler = SchedulerFactory.get_scheduler(optimizer, **scheduler_paras)
    train_full(model, train_loader, val_loader, optimizer,bos_len,scheduler, loss_func, n_epochs, device, saving_root)

__main__()





# train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# for src, trg in train_loader:
#     # push data to GPU (if available)
#     src, trg = src.to(device), trg.to(device)
#     # get model predictions
#     break
#
# from models.transformer.transformer import Transformer
# model = Transformer(src_len_max=2000, trg_len_max=2000, src_pad_idx=1, trg_pad_idx=1, d_src=1, d_trg=1, d_model=64,
#                     trg_size=1).to(device)
# y_hat = model(src, trg)