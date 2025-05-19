from data.dataset import DT_Dataset, pre_train_Dataset, pre_val_Dataset
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
import torch
from torch import nn
from utils.metrics import cal_nse_torch
# from pretrain import saving_root
from utils.lr_strategies import SchedulerFactory


def collate_fn(batch):
    # 假设每个样本是一个元组 (features, label)，其中 features 是 numpy.ndarray 类型
    features, labels = zip(*batch)
    # 将 numpy.ndarray 转换为 Tensor
    features = [torch.tensor(feature) for feature in features]
    labels = [torch.tensor(label) for label in labels]  # 假设标签已经是 Tensor 类型
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
# def loss_func(y_hat, trg, bos_len=3, pad_idx=1, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, sigma=1.0, threshold=1.0):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     mask = (trg != pad_idx)
#     y_hat = y_hat.squeeze(-1)
#
#     # Adjusting pred and ground_truth for alignment
#     y_hat_masked = y_hat.masked_fill(~mask, 0)
#     trg_masked = trg.masked_fill(~mask, 0)
#
#     # Assuming time dimension is the first dimension (seq_len, batch_size)
#     pred = y_hat_masked[bos_len - 1:-1, :]  # (seq_len_new, batch_size)
#     ground_truth = trg_masked[bos_len:, :]  # (seq_len_new, batch_size)
#
#     # Permute to (batch_size, seq_len_new) for easier processing
#     pred = pred.permute(1, 0)
#     ground_truth = ground_truth.permute(1, 0)
#
#     # MSE Loss
#     # mse_loss = nn.MSELoss()(pred, ground_truth)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     cal_loss = nn.MSELoss().to(device)
#     mask = (trg != pad_idx)
#     y_hat = y_hat.squeeze(-1)
#     pred = y_hat.masked_fill(mask == False, 0)[:][bos_len - 1:-1] if (y_hat.shape[0] > 1) else \
#     y_hat.masked_fill(mask == False, 0)[0][bos_len - 1:-1]
#     ground_truth = trg.masked_fill(mask == False, 0)[:][bos_len:] if (trg.shape[0] > 1) else \
#     trg.masked_fill(mask == False, 0)[0][bos_len:]
#     mse_loss = cal_loss(pred, ground_truth)
#     # Detect peaks for ground_truth and pred
#     def detect_peaks(data, mask):
#         seq_len = data.size(1)
#         left = data[:, :-2]
#         center = data[:, 1:-1]
#         right = data[:, 2:]
#         peak_cond = (center > left) & (center > right)
#
#         mask_left = mask[:, :-2]
#         mask_center = mask[:, 1:-1]
#         mask_right = mask[:, 2:]
#         valid = mask_left & mask_center & mask_right
#
#         peak_mask = torch.zeros_like(mask, dtype=torch.bool)
#         peak_mask[:, 1:-1] = peak_cond & valid
#         return peak_mask
#
#     mask_gt = (ground_truth != 0)
#     peak_mask_true = detect_peaks(ground_truth, mask_gt)
#
#     mask_pred = (pred != 0)
#     peak_mask_pred = detect_peaks(pred, mask_pred)
#
#     # Peak MAE
#     pred_peaks = pred[peak_mask_true]
#     true_peaks = ground_truth[peak_mask_true]
#     if pred_peaks.numel() > 0:
#         peak_mae = torch.mean(torch.abs(pred_peaks - true_peaks))
#     else:
#         peak_mae = torch.tensor(0.0, device=device)
#
#     # Time Shift Loss
#     def get_peak_indices(peak_mask):
#         batch_size, seq_len = peak_mask.size()
#         peak_indices = []
#         for i in range(batch_size):
#             indices = torch.where(peak_mask[i])[0]
#             if len(indices) == 0:
#                 peak_indices.append(torch.full((1,), -1, device=device))
#             else:
#                 peak_indices.append(indices)
#         if peak_indices:
#             max_len = max(len(p) for p in peak_indices)
#         else:
#             max_len = 0
#         padded = torch.full((batch_size, max_len), -1, device=device)
#         for i, p in enumerate(peak_indices):
#             if p[0] != -1:
#                 padded[i, :len(p)] = p
#         return padded
#
#     true_indices = get_peak_indices(peak_mask_true)
#     pred_indices = get_peak_indices(peak_mask_pred)
#     seq_len_new = ground_truth.size(1)
#     pred_indices_valid = torch.where(pred_indices == -1, seq_len_new, pred_indices)
#
#     # 张量不为空
#     if true_indices.numel() > 0:
#
#         # Compute minimum time differences
#         true_exp = true_indices.unsqueeze(2)
#         pred_exp = pred_indices_valid.unsqueeze(1)
#         diffs = torch.abs(true_exp - pred_exp)
#         min_diffs, _ = torch.min(diffs, dim=2)
#
#         valid_true_mask = (true_indices != -1)
#         min_diffs = min_diffs * valid_true_mask.float()
#         # Gaussian weighted time shift loss
#         time_shift = 1 - torch.exp(-min_diffs ** 2 / (2 * sigma ** 2))
#         num_true = valid_true_mask.sum(dim=1).clamp(min=1e-8)
#         time_shift_loss = (time_shift.sum(dim=1) / num_true).mean()
#         # MultiPeak Penalty
#         matched_true = (min_diffs <= threshold) & valid_true_mask
#         num_matched_true = matched_true.sum(dim=1)
#         num_true_peaks = valid_true_mask.sum(dim=1)
#         missed = num_true_peaks - num_matched_true
#
#         pred_exp_v2 = pred_indices.unsqueeze(2)
#         true_exp_v2 = true_indices.unsqueeze(1)
#         diffs_pred = torch.abs(pred_exp_v2 - true_exp_v2)
#         min_diffs_pred, _ = torch.min(diffs_pred, dim=2)
#
#         valid_pred_mask = (pred_indices != -1)
#         matched_pred = (min_diffs_pred <= threshold) & valid_pred_mask
#         num_matched_pred = matched_pred.sum(dim=1)
#         num_pred_peaks = valid_pred_mask.sum(dim=1)
#         false_alarm = num_pred_peaks - num_matched_pred
#
#         multipeak_penalty = (gamma * missed + delta * false_alarm).float().mean()
#     else:
#         time_shift_loss = torch.tensor(0.0, device=device)
#         multipeak_penalty = torch.tensor(0.0, device=device)
#
#
#
#     # Total loss
#     total_loss = alpha * mse_loss + beta * peak_mae + gamma * time_shift_loss + delta * multipeak_penalty
#
#     return total_loss
def __main__():
    from models.transformer.transformer import Transformer
    from utils.train_full import train_full
    from utils.metrics import loss_func_NZ
    # 设置随机种子
    seed = 42
    set_seed(seed)
    max_len = 2000
    pad_idx = 1
    bos_len = 3
    learning_rate = 1e-2
    n_epochs = 500
    d_model = 64
    out_fall_name = 'JHHN2'
    Rainfall_intensity = 'None'
    batch_size = 100
    from data.dataset import DT_Dataset, pre_train_Dataset, pre_val_Dataset
    DS_t = pre_train_Dataset
    ds_train = DS_t(outfall_name=out_fall_name)
    # DATASET INCLUDE ALL DATA
    from data.dataset import DT_Dataset_all
    from torch.utils.data import DataLoader
    from utils.collate_fn import collate_fn
    # _________dataset________
    out_fall_name = 'JHHHHC'
    max_patience = 12
    batch_size = 100
    ds_val = DT_Dataset_all(outfall_name=out_fall_name, max_patience=max_patience, type='val')
    train_loader = DataLoader(dataset=ds_train, batch_size=batch_size, shuffle=True,collate_fn=collate_fn)
    val_loader = DataLoader(dataset=ds_val, batch_size=batch_size, shuffle=False,collate_fn=collate_fn)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Transformer(src_len_max = max_len,trg_len_max=max_len, src_pad_idx = pad_idx, trg_pad_idx=pad_idx, d_src=1, d_trg=1, d_model=d_model, trg_size=1).to(device)
    # 如果有多个 GPU，使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler_paras = {"scheduler_type": "warm_up", "warm_up_epochs": n_epochs * 0.25, "decay_rate": 0.99}
    scheduler = SchedulerFactory.get_scheduler(optimizer, **scheduler_paras)
    loss_func = loss_func_NZ
    saving_root = Path(
        "./runs") / f'PRE_ml_{max_len}_padidx_{pad_idx}_bos_len_{bos_len}_lr_{learning_rate}_epochs_{n_epochs}_d_model_{d_model}_of_{out_fall_name}_RI_{Rainfall_intensity}_LF_{loss_func.__name__}_DATASET_f'
    train_full(model, train_loader, val_loader, optimizer,bos_len,scheduler, loss_func, n_epochs, device, saving_root)
__main__()