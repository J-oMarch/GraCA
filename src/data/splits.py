import torch
import numpy as np
from torch_geometric.data import Data


def generate_splits(data: Data, config: dict, seed: int = 42) -> Data:
    """Generate train/val/test splits for datasets without standard splits."""
    num_nodes = data.num_nodes
    num_classes = int(data.y.max().item()) + 1

    split_cfg = config.get("dataset", {}).get("split_config", {})
    num_train_per_class = split_cfg.get("num_train_per_class", 20)
    num_val = split_cfg.get("num_val", 500)
    num_test = split_cfg.get("num_test", 1000)

    rng = np.random.RandomState(seed)
    y_np = data.y.numpy()

    train_indices = []
    for c in range(num_classes):
        class_indices = np.where(y_np == c)[0]
        selected = rng.choice(class_indices, size=min(num_train_per_class, len(class_indices)), replace=False)
        train_indices.extend(selected.tolist())

    remaining = list(set(range(num_nodes)) - set(train_indices))
    rng.shuffle(remaining)
    val_indices = remaining[:num_val]
    test_indices = remaining[num_val:num_val + num_test]

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data
