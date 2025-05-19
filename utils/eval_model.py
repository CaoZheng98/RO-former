import torch
from torch import nn
from utils.metrics import calc_nse_torch
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

def draw_(rain_fall, pred: torch.Tensor, ground_truth: torch.Tensor, iter_, figures_folder = 'figures_jhhhhc_r10',time_step = '10 min'):
    n_step_0 = len(rain_fall[0])
    x_0 = np.arange(n_step_0)
    n_step_1 = len(pred[0])
    x_1 = np.arange(n_step_1)
    # 确保两个张量的形状相同
    assert ground_truth.shape == pred.shape

    # 将GPU张量移动到CPU并转换为NumPy数组
    ground_truth_cpu = ground_truth.cpu().numpy()
    pred_cpu = pred.cpu().numpy()
    rain_fall = rain_fall.cpu().numpy()
    # 定义保存图像的文件夹名称
    # figures_folder = 'figures___'

    # 如果文件夹不存在，则创建它
    if not os.path.exists(figures_folder):
        os.makedirs(figures_folder)

    # 基础文件名
    base_filename = f'ground_truth_vs_pred{iter_}.png'

    # 初始化后缀
    suffix = 1

    # 检查文件是否存在，如果存在则添加后缀
    filename = os.path.join(figures_folder, base_filename)
    while os.path.exists(filename):
        filename = os.path.join(figures_folder, f"{os.path.splitext(base_filename)[0]}_{suffix}{os.path.splitext(base_filename)[1]}")
        suffix += 1

    # 绘制两个张量的数据
    # plt.figure(figsize=(10, 5))
    # plt.plot(ground_truth_cpu[0, :, 0], label='Ground Truth')
    # plt.plot(pred_cpu[0, :, 0], label='Prediction', linestyle='--')
    # 创建一个图形和一个子图
    # 添加图例
    # plt.legend()
    #
    # # 添加标题和轴标签
    # plt.title('Ground Truth vs Prediction')
    # plt.xlabel('Sample Index')
    # plt.ylabel('Value')
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 绘制从上至下的条形图
    ax1.bar(x_0, rain_fall[0, :, 0], color='blue', label='rain_fall')
    ax1.set_ylabel('rain_fall(mm/min)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    # 调整y轴的刻度方向为从上至下
    ax1.set_ylim(ax1.get_ylim()[::-1])

    # 创建第二个y轴
    ax2 = ax1.twinx()

    # 绘制从下至上的曲线图
    ax2.plot(x_1, ground_truth_cpu[0, :, 0], color='red', marker='o', label='Ground Truth')
    ax2.set_ylabel('out_fall(CMS)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.legend(loc='upper right')
    # 绘制从下至上的第二条曲线图
    ax2.plot(x_1, pred_cpu[0, :, 0], color='green', marker='x', label='Prediction')
    ax2.legend(loc='upper right')  # 更新图例

    # 设置x轴标签
    ax1.set_xlabel(f'time, step:{time_step}')

    # 调整布局
    plt.tight_layout()
    # 保存图表
    plt.savefig(filename)
    plt.close()  # 关闭图表以避免重叠

    print(f"Image saved as {filename}")

def eval_model(model, data_loader, bos_len, device,figure_path = False):
    """
    Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param data_loader: A PyTorch DataLoader, providing the data.
    :param decode_mode: autoregressive or non-autoregressive
    :param device: device for data and models

    :return: mse_mean, nse_mean
    """
    # set model to eval mode (important for dropout)
    model.eval()
    mse = nn.MSELoss()
    cnt = 0
    mse_mean = 0
    nse_mean = 0
    total_batches = len(data_loader)
    current_progress = 0
    with torch.no_grad():
        for x_seq, y_seq, real_data in data_loader:
            #y_seq做了padding
            if x_seq.ndim == 2:
                x_seq = x_seq.unsqueeze(-1)
            if y_seq.ndim == 2:
                y_seq = y_seq.unsqueeze(-1)
            if real_data.ndim == 2:
                real_data = real_data.unsqueeze(-1)
            x_seq, y_seq, real_data = x_seq.to(device), y_seq.to(device),real_data.to(device)
            enc_inputs = x_seq
            batch_size = y_seq.size(0)
            trg_len = y_seq.size(1)
            n_labels = y_seq.size(2)
            dec_inputs = torch.zeros((batch_size, trg_len,n_labels)).to(device)
            dec_inputs[:,:bos_len,:] = y_seq[:,:bos_len,:]

            # get model predictions
            for i in range(bos_len, trg_len):
                decoder_predict = model(enc_inputs, dec_inputs)
                dec_inputs[:, i, :] = decoder_predict[:, i - 1, :]
            y_hat = dec_inputs[:, -(trg_len - bos_len):, :]
            y_truth = real_data[:, -(trg_len - bos_len):, :]

            # calculate loss
            mse_value = mse(y_hat, y_truth).item()
            nse_value = calc_nse_torch(y_hat, y_truth)
            cnt += 1
            mse_mean = mse_mean + (mse_value - mse_mean) / cnt  # Welford’s method
            nse_mean = nse_mean + (nse_value - nse_mean) / cnt  # Welford’s method
            # 数据可视化：
            progress = (cnt / total_batches) * 100
            if progress >= current_progress + 90:
                draw_(x_seq, y_hat, y_truth, current_progress, figures_folder=figure_path)
                # 更新当前进度
                current_progress = progress
    return mse_mean, nse_mean


def eval_model_obs_preds(model, data_loader, decode_mode, device):
    """
    Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param data_loader: A PyTorch DataLoader, providing the data.
    :param decode_mode: autoregressive or non-autoregressive
    :param device: device for data and models

    :return: Two torch Tensors, containing the observations and model predictions
    """
    # set model to eval mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    with torch.no_grad():
        for x_seq, y_seq_past, y_seq_future, _ in data_loader:
            x_seq, y_seq_past, y_seq_future = x_seq.to(device), y_seq_past.to(device), y_seq_future.to(device)
            batch_size = y_seq_past.shape[0]
            tgt_len = y_seq_past.shape[1] + y_seq_future.shape[1]
            tgt_size = y_seq_future.shape[2]
            pred_len = y_seq_future.shape[1]

            enc_inputs = x_seq
            dec_inputs = torch.zeros((batch_size, tgt_len, tgt_size)).to(device)
            dec_inputs[:, :-pred_len, :] = y_seq_past
            # get model predictions
            if decode_mode == "NAR":
                y_hat = model(enc_inputs, dec_inputs)
                y_hat = y_hat[:, -pred_len:, :]
            elif decode_mode == "AR":
                for i in range(tgt_len - pred_len, tgt_len):
                    decoder_predict = model(enc_inputs, dec_inputs)
                    dec_inputs[:, i, :] = decoder_predict[:, i - 1, :]
                y_hat = dec_inputs[:, -pred_len:, :]
            else:  # Model is not Transformer
                y_hat = model(x_seq, y_seq_past)

            obs.append(y_seq_future.to("cpu"))
            preds.append(y_hat.to("cpu"))

    obs_all = torch.cat(obs)
    preds_all = torch.cat(preds)

    return obs_all, preds_all
