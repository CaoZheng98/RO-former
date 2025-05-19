import torch
import time
import os
from utils.train_epoch import train_epoch
from utils.eval_model import eval_model
from utils.tools import BoardWriter, count_parameters
from pathlib import Path
import torch

import os
from pathlib import Path
import torch

class BestModelLog:
    def __init__(self, init_model, optimizer, scheduler, saving_root, metric_name, high_better: bool):
        self.high_better = high_better
        self.saving_root = saving_root
        self.metric_name = metric_name
        self.optimizer = optimizer
        self.scheduler = scheduler

        # 初始化最佳记录
        worst = float("-inf") if high_better else float("inf")
        self.best_epoch = -1
        self.best_value = worst
        self.best_checkpoint_path = self.saving_root / f"best_({metric_name})_{self.best_epoch}_{self.best_value}.pt"

        # 保存初始状态
        self.save_checkpoint(init_model, epoch=0, value=worst)

    def save_checkpoint(self, model, epoch, value):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metric_value": value
        }
        torch.save(checkpoint, self.best_checkpoint_path)

    def update(self, model, new_value, epoch):
        if ((self.high_better and (new_value > self.best_value)) or
            (not self.high_better and (new_value < self.best_value))):

            # 删除旧的最佳模型文件
            if os.path.exists(self.best_checkpoint_path):
                os.remove(self.best_checkpoint_path)

            # 更新最佳记录
            self.best_value = new_value
            self.best_epoch = epoch
            self.best_checkpoint_path = self.saving_root / f"best_({self.metric_name})_{self.best_epoch}_{self.best_value:.4f}.pt"

            # 保存完整 checkpoint
            self.save_checkpoint(model, epoch, new_value)


def train_full_evaluated_by_train(model, train_loader, val_loader, optimizer, bos_len, scheduler, loss_func, n_epochs, device,
               saving_root, using_board=False, early_stop_patience=200):
    print(f"Parameters count:{count_parameters(model)}")
    log_file = saving_root / "log_train.csv"
    figure_path = Path(f'./{saving_root.name}')
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)
    with open(log_file, "wt") as f:
        f.write(f"parameters_count:{count_parameters(model)}\n")
        f.write("epoch,train_loss_iterated,train_mse,train_nse\n")
    if using_board:
        tb_root = saving_root / "tb_log"
        writer = BoardWriter(tb_root)
    else:
        writer = None
    mse_log = BestModelLog(model, optimizer, scheduler, saving_root, "min_mse", high_better=False)
    nse_log = BestModelLog(model, optimizer, scheduler, saving_root, "max_nse", high_better=True)
    newest_log = BestModelLog(model, optimizer, scheduler, saving_root, "newest", high_better=True)

    t1 = time.time()
    # Early stopping initialization
    if early_stop_patience is not None:
        best_nse = -1 * float('inf')
        early_stop_counter = 0
    for i in range(n_epochs):
        print(f"Training progress: {i} / {n_epochs}")
        train_loss_iterated = train_epoch(model, train_loader, optimizer, bos_len, scheduler, loss_func, device, i)
        mse_train, nse_train = '', ''
        # 注释下一行可加速训练
        mse_train, nse_train = eval_model(model, train_loader, bos_len, device,figure_path=figure_path)
        # mse_val, nse_val = eval_model(model, val_loader, bos_len, device,figure_path=figure_path)

        if writer is not None:
            if (mse_train != '') and (nse_train != ''):
                writer.write_board(f"train_mse", metric_value=mse_train, epoch=i)
                writer.write_board(f"train_nse", metric_value=nse_train, epoch=i)
            writer.write_board(f"train_loss(iterated)", metric_value=train_loss_iterated, epoch=i)
            # writer.write_board("val_mse", metric_value=mse_val, epoch=i)
            # writer.write_board("val_nse", metric_value=nse_val, epoch=i)
        with open(log_file, "at") as f:
            f.write(f"{i},{train_loss_iterated},{mse_train},{nse_train}\n")
        mse_log.update(model, mse_train, i)
        nse_log.update(model, nse_train, i)
        newest_log.update(model, i, i)

        # Early stopping check
        if early_stop_patience is not None:
            if nse_train > best_nse:
                best_nse = nse_train
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {i}. Best validation NSE: {best_nse:.4f}")
                    break  # 退出训练循环
    t2 = time.time()
    print(f"Training used time:{t2 - t1}")
    print(f"train_loss_iterated:{train_loss_iterated}")

def train_full(model, train_loader, val_loader, optimizer, bos_len, scheduler, loss_func, n_epochs, device,
               saving_root, using_board=False, early_stop_patience=2020):
    print(f"Parameters count:{count_parameters(model)}")
    log_file = saving_root / "log_train.csv"
    figure_path = Path(f'./{saving_root.name}')
    if not os.path.exists(saving_root):
        os.makedirs(saving_root)
    with open(log_file, "wt") as f:
        f.write(f"parameters_count:{count_parameters(model)}\n")
        f.write("epoch,train_loss_iterated,val_mse,val_nse\n")
    if using_board:
        tb_root = saving_root / "tb_log"
        writer = BoardWriter(tb_root)
    else:
        writer = None
    mse_log = BestModelLog(model, optimizer, scheduler, saving_root, "min_mse", high_better=False)
    nse_log = BestModelLog(model, optimizer, scheduler, saving_root, "max_nse", high_better=True)
    newest_log = BestModelLog(model, optimizer, scheduler, saving_root, "newest", high_better=True)

    t1 = time.time()
    # Early stopping initialization
    if early_stop_patience is not None:
        best_nse = -1 * float('inf')
        early_stop_counter = 0
    for i in range(n_epochs):
        print(f"Training progress: {i} / {n_epochs}")
        train_loss_iterated = train_epoch(model, train_loader, optimizer, bos_len, scheduler, loss_func, device, i)
        mse_train, nse_train = '', ''
        # 注释下一行可加速训练
        # mse_train, nse_train = eval_model(model, train_loader, bos_len, device,figure_path=figure_path)
        mse_val, nse_val = eval_model(model, val_loader, bos_len, device,figure_path=figure_path)

        if writer is not None:
            if (mse_train != '') and (nse_train != ''):
                writer.write_board(f"train_mse", metric_value=mse_train, epoch=i)
                writer.write_board(f"train_nse", metric_value=nse_train, epoch=i)
            writer.write_board(f"train_loss(iterated)", metric_value=train_loss_iterated, epoch=i)
            writer.write_board("val_mse", metric_value=mse_val, epoch=i)
            writer.write_board("val_nse", metric_value=nse_val, epoch=i)
        with open(log_file, "at") as f:
            f.write(f"{i},{train_loss_iterated},{mse_val},{nse_val}\n")
        mse_log.update(model, mse_val, i)
        nse_log.update(model, nse_val, i)
        newest_log.update(model, i, i)

        # Early stopping check
        if early_stop_patience is not None:
            if nse_val > best_nse:
                best_nse = nse_val
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stop_patience:
                    print(f"Early stopping triggered at epoch {i}. Best validation NSE: {best_nse:.4f}")
                    break  # 退出训练循环
    t2 = time.time()
    print(f"Training used time:{t2 - t1}")
    print(f"train_loss_iterated:{train_loss_iterated}")

# def train_full(model, train_loader, val_loader, optimizer, bos_len, scheduler, loss_func, n_epochs, device,
#                saving_root, using_board=False):
#     print(f"Parameters count:{count_parameters(model)}")
#     log_file = saving_root / "log_train.csv"
#     if not os.path.exists(saving_root):
#         os.makedirs(saving_root)
#     with open(log_file, "wt") as f:
#         f.write(f"parameters_count:{count_parameters(model)}\n")
#         f.write("epoch,train_loss_iterated,train_mse,train_nse\n")
#     if using_board:
#         tb_root = saving_root / "tb_log"
#         writer = BoardWriter(tb_root)
#     else:
#         writer = None
#
#     mse_log = BestModelLog(model, saving_root, "min_mse", high_better=False)
#     nse_log = BestModelLog(model, saving_root, "max_nse", high_better=True)
#     newest_log = BestModelLog(model, saving_root, "newest", high_better=True)
#
#     t1 = time.time()
#     for i in range(n_epochs):
#         print(f"Training progress: {i} / {n_epochs}")
#         train_loss_iterated = train_epoch(model, train_loader, optimizer, bos_len, scheduler, loss_func, device, i)
#         mse_train, nse_train = '', ''
#         # mse_train, nse_train is not need to be calculated (via eval_model function),
#         # and you can comment the next line to speed up
#         mse_train, nse_train = eval_model(model, train_loader, bos_len, device)
#         mse_val, nse_val = eval_model(model, val_loader, bos_len, device)
#         if writer is not None:
#             if (mse_train != '') and (nse_train != ''):
#                 writer.write_board(f"train_mse", metric_value=mse_train, epoch=i)
#                 writer.write_board(f"train_nse", metric_value=nse_train, epoch=i)
#             writer.write_board(f"train_loss(iterated)", metric_value=train_loss_iterated, epoch=i)
#             writer.write_board("val_mse", metric_value=mse_val, epoch=i)
#             writer.write_board("val_nse", metric_value=nse_val, epoch=i)
#         with open(log_file, "at") as f:
#             f.write(f"{i},{train_loss_iterated},{mse_train},{nse_train}\n")
#         mse_log.update(model, mse_val, i)
#         nse_log.update(model, nse_val, i)
#         newest_log.update(model, i, i)
#     t2 = time.time()
#     print(f"Training used time:{t2 - t1}")
#     print(f"train_loss_iterated:{train_loss_iterated}")
