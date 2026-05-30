from src.training.train_downstream import train_downstream


def run_original(data, config, num_features, num_classes, device, seed=42):
    """Original baseline: train downstream on the original graph."""
    results = {}
    downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
    for model_name in downstream_names:
        results[model_name] = train_downstream(
            model_name=model_name,
            data=data,
            edge_index=data.edge_index,
            config=config,
            num_features=num_features,
            num_classes=num_classes,
            device=device,
            seed=seed,
        )
    return results
