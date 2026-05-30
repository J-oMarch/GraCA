import torch


def get_device(config: dict) -> torch.device:
    """Resolve device from config string."""
    device_str = config.get("training", {}).get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_str)
