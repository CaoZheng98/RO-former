import torch
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