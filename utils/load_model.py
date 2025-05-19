from models.transformer.transformer import Transformer  # 替换为你的模型模块路径
import torch
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