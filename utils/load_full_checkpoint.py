import torch
def load_full_checkpoint(checkpoint_path, model, optimizer, scheduler, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)  # 注意安全风险

    # 加载模型参数
    model.load_state_dict(checkpoint["model_state_dict"])

    # 加载优化器和调度器状态
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # 返回断点 epoch 和指标值
    return {
        "epoch": checkpoint["epoch"],
        "metric_value": checkpoint["metric_value"]
    }
