"""Original stratified fixed-test and repeated train/validation splits."""
import numpy as np
import torch as th

def split_fixed_test_stratified(labels, test_ratio=0.2, seed=42):
    """
    对每个类别分层抽样，选出 test_ratio 的样本作为测试集。
    返回：idx_test, idx_remaining
    """
    np.random.seed(seed)
    labels_np = labels.cpu().numpy() if isinstance(labels, th.Tensor) else labels
    unique_classes = np.unique(labels_np)
    test_indices = []
    remaining_indices = []
    for cls in unique_classes:
        cls_indices = np.where(labels_np == cls)[0]
        np.random.shuffle(cls_indices)
        n_test = int(len(cls_indices) * test_ratio)
        test_indices.extend(cls_indices[:n_test])
        remaining_indices.extend(cls_indices[n_test:])
    return (th.LongTensor(test_indices), th.LongTensor(remaining_indices))

def random_train_val_split_stratified(remaining_indices, labels, train_ratio=0.75, seed=None):
    """
    对剩余样本进行分层随机划分为 train/val。
    """
    if seed is not None:
        np.random.seed(seed)
    labels_np = labels.cpu().numpy() if isinstance(labels, th.Tensor) else labels
    remaining_np = remaining_indices.cpu().numpy() if isinstance(remaining_indices, th.Tensor) else remaining_indices
    unique_classes = np.unique(labels_np)
    idx_train = []
    idx_val = []
    for cls in unique_classes:
        cls_indices = np.where(labels_np[remaining_np] == cls)[0]
        cls_remaining = np.array(remaining_np)[cls_indices]
        np.random.shuffle(cls_remaining)
        n_train = int(len(cls_remaining) * train_ratio)
        idx_train.extend(cls_remaining[:n_train])
        idx_val.extend(cls_remaining[n_train:])
    return (th.LongTensor(idx_train), th.LongTensor(idx_val))

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
