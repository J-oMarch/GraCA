"""
Train-internal support/score split for GraGE.

Splits train_mask into support_mask and score_mask:
- support_mask: used for inner-loop training (gradient descent steps)
- score_mask: used for outer-loop loss computation (hypergradient target)
- support_mask | score_mask == train_mask
- support_mask & score_mask == empty

No val/test labels are used.
"""
import torch
import numpy as np
from typing import Tuple


def split_train_support_score(
    train_mask: torch.Tensor,
    y: torch.Tensor = None,
    score_ratio: float = 0.3,
    seed: int = 0,
    stratified: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split train_mask into support_mask and score_mask.

    Args:
        train_mask: [N] boolean mask for training nodes
        y: [N] node labels (required if stratified=True)
        score_ratio: fraction of train nodes to use for scoring
        seed: random seed for reproducibility
        stratified: if True, preserve class distribution in both splits

    Returns:
        support_mask: [N] boolean mask for support (inner-loop) training
        score_mask: [N] boolean mask for score (outer-loop) loss computation

    Requirements:
        - support_mask & score_mask == empty (disjoint)
        - support_mask | score_mask == train_mask (exhaustive)
        - No val/test labels used
        - Each class has at least 1 sample in both splits (if enough samples)
    """
    train_indices = torch.where(train_mask)[0].cpu()
    n_train = len(train_indices)

    if n_train == 0:
        return train_mask.clone(), train_mask.clone()

    n_score = max(1, int(n_train * score_ratio))
    n_support = n_train - n_score

    # Ensure minimum sizes
    if n_support < 1:
        n_support = 1
        n_score = n_train - 1
    if n_score < 1:
        n_score = 1
        n_support = n_train - 1

    rng = np.random.RandomState(seed)

    if stratified and y is not None:
        # Stratified split preserving class distribution
        y_train = y[train_mask].cpu().numpy()
        classes = np.unique(y_train)

        score_indices = []
        support_indices = []

        for c in classes:
            class_mask = y_train == c
            class_indices = train_indices[class_mask].numpy()
            n_class = len(class_indices)

            if n_class <= 2:
                # Too few samples: put 1 in each split
                perm = rng.permutation(n_class)
                score_indices.append(class_indices[perm[:1]])
                support_indices.append(class_indices[perm[1:]])
            else:
                # Proportional split
                n_c_score = max(1, int(n_class * score_ratio))
                n_c_support = n_class - n_c_score

                # Ensure both have at least 1
                if n_c_support < 1:
                    n_c_support = 1
                    n_c_score = n_class - 1
                if n_c_score < 1:
                    n_c_score = 1
                    n_c_support = n_class - 1

                perm = rng.permutation(n_class)
                score_indices.append(class_indices[perm[:n_c_score]])
                support_indices.append(class_indices[perm[n_c_score:]])

        score_indices = np.concatenate(score_indices)
        support_indices = np.concatenate(support_indices)
    else:
        # Random split (not stratified)
        perm = rng.permutation(n_train)
        score_indices = train_indices[perm[:n_score]].numpy()
        support_indices = train_indices[perm[n_score:]].numpy()

    # Build masks
    support_mask = torch.zeros_like(train_mask)
    score_mask = torch.zeros_like(train_mask)

    support_mask[support_indices] = True
    score_mask[score_indices] = True

    # Verify correctness
    assert (support_mask & score_mask).sum() == 0, "support and score masks must be disjoint"
    assert (support_mask | score_mask).sum() == train_mask.sum(), \
        "support | score must equal train_mask"

    return support_mask, score_mask
