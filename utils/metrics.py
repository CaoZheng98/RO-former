# All metrics needs obs and sim shape: (batch_size, pred_len, tgt_size)

import numpy as np
import torch
import torch.nn as nn

def calc_nse(obs: np.array, sim: np.array) -> np.array:
    denominator = np.sum((obs - np.mean(obs, axis=0)) ** 2, axis=0)
    numerator = np.sum((sim - obs) ** 2, axis=0)
    nse = 1 - numerator / denominator

    nse_mean = np.mean(nse)
    # Return mean NSE, and NSE of all locations, respectively
    return nse_mean, nse[:, 0]


def calc_kge(obs: np.array, sim: np.array):
    mean_obs = np.mean(obs, axis=0)
    mean_sim = np.mean(sim, axis=0)

    std_obs = np.std(obs, axis=0)
    std_sim = np.std(sim, axis=0)

    beta = mean_sim / mean_obs
    alpha = std_sim / std_obs
    numerator = np.mean(((obs - mean_obs) * (sim - mean_sim)), axis=0)
    denominator = std_obs * std_sim
    gamma = numerator / denominator
    kge = 1 - np.sqrt((beta - 1) ** 2 + (alpha - 1) ** 2 + (gamma - 1) ** 2)

    kge_mean = np.mean(kge)
    # Return mean KEG, and KGE of all locations, respectively
    return kge_mean, kge[:, 0]


def calc_tpe(obs: np.array, sim: np.array, alpha):
    sort_index = np.argsort(obs, axis=0)
    obs_sort = np.take_along_axis(obs, sort_index, axis=0)
    sim_sort = np.take_along_axis(sim, sort_index, axis=0)
    top = int(obs.shape[0] * alpha)
    obs_t = obs_sort[-top:, :]
    sim_t = sim_sort[-top:, :]
    numerator = np.sum(np.abs(sim_t - obs_t), axis=0)
    denominator = np.sum(obs_t, axis=0)
    tpe = numerator / denominator

    tpe_mean = np.mean(tpe)
    # Return mean TPE, and TPE of all locations, respectively
    return tpe_mean, tpe[:, 0]


def calc_bias(obs: np.array, sim: np.array):
    numerator = np.sum(sim - obs, axis=0)
    denominator = np.sum(obs, axis=0)
    bias = numerator / denominator

    bias_mean = np.mean(bias)
    # Return mean bias, and bias of all locations, respectively
    return bias_mean, bias[:, 0]


def calc_mse(obs: np.array, sim: np.array):
    mse = np.mean((obs - sim) ** 2, axis=0)

    mse_mean = np.mean(mse)
    # Return mean MSE, and MSE of all locations, respectively
    return mse_mean, mse[:, 0]


def cacl_rmse(obs: np.array, sim: np.array):
    mse = np.mean((obs - sim) ** 2, axis=0)
    rmse = np.sqrt(mse)

    rmse_mean = np.mean(rmse)
    # Return mean RMSE, and RMSE of all locations, respectively
    return rmse_mean, rmse[:, 0]


def cacl_nrmse(obs: np.array, sim: np.array):
    mse = np.mean((obs - sim) ** 2, axis=0)
    rmse = np.sqrt(mse)
    obs_mean = np.mean(obs, axis=0)
    nrmse = rmse / obs_mean

    nrmse_mean = np.mean(nrmse)
    # Return mean NRMSE, and NRMSE of all locations, respectively
    return nrmse_mean, nrmse[:, 0]

def cal_nse_torch(obs, sim):
    # denominator = torch.sum((obs - torch.mean(obs, dim=0)) ** 2, dim=0)
    # numerator = torch.sum((sim - obs) ** 2, dim=0)
    # nse = torch.tensor(1).to(sim.device) - numerator / denominator
    #
    # nse_mean = torch.mean(nse)
    # # Return mean NSE, and NSE of all locations, respectively
    eps = 0.00001
    losssum = 0
    nsample = 0
    Ngage = obs.shape[0]
    for ii in range(Ngage):
        p0 = sim[ii, :, 0]
        t0 = obs[ii, :, 0]
        mask = t0 == t0
        if len(mask[mask == True]) > 0:
            p = p0[mask]
            t = t0[mask]
            tmean = t.mean()
            SST = torch.sum((t - tmean) ** 2) + eps
            if SST != 0:
                SSRes = torch.sum((t - p) ** 2)
                temp = 1 - SSRes / SST
                losssum = losssum + temp
                nsample = nsample + 1
    # minimize the opposite average NSE
    nse_mean = losssum / nsample
    return nse_mean
def calc_nse_torch(obs, sim):
    with torch.no_grad():
        # denominator = torch.sum((obs - torch.mean(obs, dim=0)) ** 2, dim=0)
        # numerator = torch.sum((sim - obs) ** 2, dim=0)
        # nse = torch.tensor(1).to(sim.device) - numerator / denominator
        #
        # nse_mean = torch.mean(nse)
        # # Return mean NSE, and NSE of all locations, respectively
        eps = 0.00001
        losssum = 0
        nsample = 0
        Ngage = obs.shape[0]
        for ii in range(Ngage):
            p0 = sim[ii, :, 0]
            t0 = obs[ii, :, 0]
            mask = t0 == t0
            if len(mask[mask == True]) > 0:
                p = p0[mask]
                t = t0[mask]
                tmean = t.mean()
                SST = torch.sum((t - tmean) ** 2) + eps
                if SST != 0:
                    SSRes = torch.sum((t - p) ** 2)
                    temp = 1 - SSRes / SST
                    losssum = losssum + temp
                    nsample = nsample + 1
        # minimize the opposite average NSE
        nse_mean = losssum / nsample
        return nse_mean
    # with torch.no_grad():
    #     denominator = torch.sum((obs - torch.mean(obs, dim=0)) ** 2, dim=0)
    #     numerator = torch.sum((sim - obs) ** 2, dim=0)
    #     nse = torch.tensor(1).to(sim.device) - numerator / denominator
    #
    #     nse_mean = torch.mean(nse)
    #     # Return mean NSE, and NSE of all locations, respectively
    #     return nse_mean, nse[:, 0]
def loss_func_mse(y_hat, trg, bos_len=3,pad_idx=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cal_loss = nn.MSELoss().to(device)
    mask = (trg != pad_idx)
    y_hat = y_hat.squeeze(-1)
    pred = y_hat.masked_fill(mask == False, 0)[:][bos_len-1:-1] if (y_hat.shape[0]>1) else y_hat.masked_fill(mask == False, 0)[0][bos_len-1:-1]
    ground_truth = trg.masked_fill(mask == False, 0)[:][bos_len:] if (trg.shape[0]>1) else trg.masked_fill(mask == False, 0)[0][bos_len:]

    loss = cal_loss(pred, ground_truth)
    return loss

def loss_func_NSE_enhanced(y_hat, trg, bos_len=1,pad_idx=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cal_loss = nn.MSELoss
    cal_loss = cal_nse_torch
    # 创建掩码
    mask = (trg != pad_idx)
    zero = (trg != 0)
    # 压缩 y_hat 的最后一个维度
    y_hat = y_hat.squeeze(-1)
    # 对 y_hat 应用掩码
    pred = y_hat.masked_fill(~mask, 0)
    pred0 = pred[:, bos_len - 1:-1]
    # 对 pred 应用 zero 掩码
    pred = pred.masked_fill(~zero, 0)
    # 切片操作
    pred = pred[:, bos_len - 1:-1]  # 确保切片范围正确

    # 对 trg 应用掩码
    ground_truth = trg.masked_fill(~mask, 0)
    ground_truth0 = ground_truth[:, bos_len:]
    # 对 ground_truth 应用 zero 掩码
    ground_truth = ground_truth.masked_fill(~zero, 0)
    # 切片操作
    ground_truth = ground_truth[:, bos_len:]  # 确保切片范围正确
    # 计算损失
    loss = -1.5 * cal_loss(pred.unsqueeze(-1), ground_truth.unsqueeze(-1)) + -1 * cal_loss(pred0.unsqueeze(-1), ground_truth0.unsqueeze(-1))
    return loss

def loss_func_nse(y_hat, trg, bos_len=1,pad_idx=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cal_loss = nn.MSELoss
    cal_loss = cal_nse_torch
    # 创建掩码
    mask = (trg != pad_idx)
    zero = (trg != 0)
    # 压缩 y_hat 的最后一个维度
    y_hat = y_hat.squeeze(-1)
    # 对 y_hat 应用掩码
    pred = y_hat.masked_fill(~mask, 0)
    pred0 = pred[:, bos_len - 1:-1]
    # 对 pred 应用 zero 掩码
    pred = pred.masked_fill(~zero, 0)
    # 切片操作
    pred = pred[:, bos_len - 1:-1]  # 确保切片范围正确

    # 对 trg 应用掩码
    ground_truth = trg.masked_fill(~mask, 0)
    ground_truth0 = ground_truth[:, bos_len:]
    # 对 ground_truth 应用 zero 掩码
    ground_truth = ground_truth.masked_fill(~zero, 0)
    # 切片操作
    ground_truth = ground_truth[:, bos_len:]  # 确保切片范围正确
    # 计算损失
    loss = -1 * cal_loss(pred0.unsqueeze(-1), ground_truth0.unsqueeze(-1))
    return loss

def loss_func_NZ(y_hat, trg, bos_len=1,pad_idx=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # cal_loss = nn.MSELoss
    cal_loss = cal_nse_torch
    cal_loss_1 = nn.MSELoss().to(device)
    # 创建掩码
    mask = (trg != pad_idx)
    zero = (trg != 0)
    # 压缩 y_hat 的最后一个维度
    y_hat = y_hat.squeeze(-1)
    # 对 y_hat 应用掩码
    pred = y_hat.masked_fill(~mask, 0)
    pred0 = pred[:, bos_len - 1:-1]
    # 对 pred 应用 zero 掩码
    pred = pred.masked_fill(~zero, 0)
    # 切片操作
    pred = pred[:, bos_len - 1:-1]  # 确保切片范围正确

    # 对 trg 应用掩码
    ground_truth = trg.masked_fill(~mask, 0)
    ground_truth0 = ground_truth[:, bos_len:]
    # 对 ground_truth 应用 zero 掩码
    ground_truth = ground_truth.masked_fill(~zero, 0)
    # 切片操作
    ground_truth = ground_truth[:, bos_len:]  # 确保切片范围正确
    # 计算损失
    loss = -1 * cal_loss(pred0.unsqueeze(-1), ground_truth0.unsqueeze(-1)) + cal_loss_1(pred0, ground_truth0)
    return loss


def loss_func_advanced(y_hat, trg, bos_len=3, pad_idx=1, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, sigma=1.0, threshold=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask = (trg != pad_idx)
    y_hat = y_hat.squeeze(-1)

    # Adjusting pred and ground_truth for alignment
    y_hat_masked = y_hat.masked_fill(~mask, 0)
    trg_masked = trg.masked_fill(~mask, 0)

    # Assuming time dimension is the first dimension (seq_len, batch_size)
    pred = y_hat_masked[bos_len - 1:-1, :]  # (seq_len_new, batch_size)
    ground_truth = trg_masked[bos_len:, :]  # (seq_len_new, batch_size)

    # Permute to (batch_size, seq_len_new) for easier processing
    pred = pred.permute(1, 0)
    ground_truth = ground_truth.permute(1, 0)

    # MSE Loss
    # mse_loss = nn.MSELoss()(pred, ground_truth)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cal_loss = nn.MSELoss().to(device)
    mask = (trg != pad_idx)
    y_hat = y_hat.squeeze(-1)
    pred = y_hat.masked_fill(mask == False, 0)[:][bos_len - 1:-1] if (y_hat.shape[0] > 1) else \
    y_hat.masked_fill(mask == False, 0)[0][bos_len - 1:-1]
    ground_truth = trg.masked_fill(mask == False, 0)[:][bos_len:] if (trg.shape[0] > 1) else \
    trg.masked_fill(mask == False, 0)[0][bos_len:]
    mse_loss = cal_loss(pred, ground_truth)
    # Detect peaks for ground_truth and pred
    def detect_peaks(data, mask):
        seq_len = data.size(1)
        left = data[:, :-2]
        center = data[:, 1:-1]
        right = data[:, 2:]
        peak_cond = (center > left) & (center > right)

        mask_left = mask[:, :-2]
        mask_center = mask[:, 1:-1]
        mask_right = mask[:, 2:]
        valid = mask_left & mask_center & mask_right

        peak_mask = torch.zeros_like(mask, dtype=torch.bool)
        peak_mask[:, 1:-1] = peak_cond & valid
        return peak_mask

    mask_gt = (ground_truth != 0)
    peak_mask_true = detect_peaks(ground_truth, mask_gt)

    mask_pred = (pred != 0)
    peak_mask_pred = detect_peaks(pred, mask_pred)

    # Peak MAE
    pred_peaks = pred[peak_mask_true]
    true_peaks = ground_truth[peak_mask_true]
    if pred_peaks.numel() > 0:
        peak_mae = torch.mean(torch.abs(pred_peaks - true_peaks))
    else:
        peak_mae = torch.tensor(0.0, device=device)

    # Time Shift Loss
    def get_peak_indices(peak_mask):
        batch_size, seq_len = peak_mask.size()
        peak_indices = []
        for i in range(batch_size):
            indices = torch.where(peak_mask[i])[0]
            if len(indices) == 0:
                peak_indices.append(torch.full((1,), -1, device=device))
            else:
                peak_indices.append(indices)
        if peak_indices:
            max_len = max(len(p) for p in peak_indices)
        else:
            max_len = 0
        padded = torch.full((batch_size, max_len), -1, device=device)
        for i, p in enumerate(peak_indices):
            if p[0] != -1:
                padded[i, :len(p)] = p
        return padded

    true_indices = get_peak_indices(peak_mask_true)
    pred_indices = get_peak_indices(peak_mask_pred)
    seq_len_new = ground_truth.size(1)
    pred_indices_valid = torch.where(pred_indices == -1, seq_len_new, pred_indices)

    # 张量不为空
    if true_indices.numel() > 0:

        # Compute minimum time differences
        true_exp = true_indices.unsqueeze(2)
        pred_exp = pred_indices_valid.unsqueeze(1)
        diffs = torch.abs(true_exp - pred_exp)
        min_diffs, _ = torch.min(diffs, dim=2)

        valid_true_mask = (true_indices != -1)
        min_diffs = min_diffs * valid_true_mask.float()
        # Gaussian weighted time shift loss
        time_shift = 1 - torch.exp(-min_diffs ** 2 / (2 * sigma ** 2))
        num_true = valid_true_mask.sum(dim=1).clamp(min=1e-8)
        time_shift_loss = (time_shift.sum(dim=1) / num_true).mean()
        # MultiPeak Penalty
        matched_true = (min_diffs <= threshold) & valid_true_mask
        num_matched_true = matched_true.sum(dim=1)
        num_true_peaks = valid_true_mask.sum(dim=1)
        missed = num_true_peaks - num_matched_true

        pred_exp_v2 = pred_indices.unsqueeze(2)
        true_exp_v2 = true_indices.unsqueeze(1)
        diffs_pred = torch.abs(pred_exp_v2 - true_exp_v2)
        min_diffs_pred, _ = torch.min(diffs_pred, dim=2)

        valid_pred_mask = (pred_indices != -1)
        matched_pred = (min_diffs_pred <= threshold) & valid_pred_mask
        num_matched_pred = matched_pred.sum(dim=1)
        num_pred_peaks = valid_pred_mask.sum(dim=1)
        false_alarm = num_pred_peaks - num_matched_pred

        multipeak_penalty = (gamma * missed + delta * false_alarm).float().mean()
    else:
        time_shift_loss = torch.tensor(0.0, device=device)
        multipeak_penalty = torch.tensor(0.0, device=device)



    # Total loss
    total_loss = alpha * mse_loss + beta * peak_mae + gamma * time_shift_loss + delta * multipeak_penalty

    return total_loss