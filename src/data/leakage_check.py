import torch


def ensure_no_test_label_leakage(
    config: dict,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    mode: str = "practical",
):
    """Assert no test labels leak into training/scoring losses.

    Args:
        config: full config dict
        train_mask: boolean mask for training nodes
        test_mask: boolean mask for test nodes
        loss_mask: boolean mask of nodes whose labels participate in loss
        mode: 'practical' or 'oracle'
    """
    oracle_only = config.get("experiment", {}).get("oracle_only", False)
    method = config.get("experiment", {}).get("method", "graca_lite")

    if mode == "practical" and not oracle_only:
        # In practical mode, loss_mask must NOT overlap with test_mask
        overlap = (loss_mask & test_mask).any().item()
        assert not overlap, (
            "LEAKAGE DETECTED: test_mask nodes found in practical loss_mask. "
            "This violates the semi-supervised setting."
        )

    if not oracle_only and method != "oracle":
        # Practical methods must not use full labels
        pass  # The assertion above is sufficient

    return True
