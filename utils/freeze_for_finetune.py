# ================== 推荐冻结方案 ==================
# 策略：冻结编码器+嵌入层，微调解码器+输出层（适合序列生成任务微调）
def freeze_for_finetune(model):
    # 冻结所有编码器参数（包含2个EncoderLayer）
    for name, param in model.named_parameters():
        if 'encoder' in name:  # 编码器相关参数
            param.requires_grad = False

    # 冻结源/目标的输入嵌入线性变换
    model.src_linear.requires_grad_(False)  # 输入特征投影层
    model.trg_linear.requires_grad_(False)

    # 冻结位置编码（注意：PositionalEncoding中的pos_table是buffer不是参数，无需处理）

    # 验证冻结情况
    print("冻结后关键参数状态：")
    check_params = [
        'encoder.layer_stack.0.slf_attn.w_qs.weight',
        'src_linear.weight',
        'decoder.layer_stack.0.slf_attn.w_qs.weight',
        'linear_project.weight'
    ]
    for name, param in model.named_parameters():
        if any(k in name for k in check_params):
            print(f"{name:50} | 可训练: {param.requires_grad}")
# ================== 推荐冻结方案2 ==================
def freeze_for_finetune_2(model):
    # 冻结所有编码器参数（包含2个EncoderLayer）
    for name, param in model.named_parameters():
        if 'encoder' in name:  # 编码器相关参数
            param.requires_grad = False

    # 冻结源/目标的输入嵌入线性变换
    model.src_linear.requires_grad_(False)  # 输入特征投影层
    model.trg_linear.requires_grad_(False)

    # 冻结解码器的第一层
    for name, param in model.named_parameters():
        if 'decoder.layer_stack.0' in name:  # 只冻结第一个DecoderLayer
            param.requires_grad = False

    # 验证冻结情况
    print("冻结后关键参数状态：")
    check_params = [
        'encoder.layer_stack.0.slf_attn.w_qs.weight',
        'src_linear.weight',
        'decoder.layer_stack.0.slf_attn.w_qs.weight',  # 应冻结
        'decoder.layer_stack.1.slf_attn.w_qs.weight',   # 应保持可训练
        'linear_project.weight'
    ]
    for name, param in model.named_parameters():
        if any(k in name for k in check_params):
            print(f"{name:50} | 可训练: {param.requires_grad}")