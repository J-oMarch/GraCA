import torch
import time
from src.models.model_factory import build_model
from src.training.losses import supervised_loss
from src.training.evaluator import evaluate
from src.training.early_stopping import EarlyStopping
from src.utils.seed import set_seed
from src.utils.logger import get_logger


def train_downstream(
    model_name: str,
    data,
    edge_index: torch.Tensor,
    config: dict,
    num_features: int,
    num_classes: int,
    device,
    seed: int = 42,
) -> dict:
    """Train a downstream GNN from scratch on a (possibly sanitized) graph.

    Args:
        model_name: 'GCN', 'GAT', or 'GraphSAGE'
        data: PyG Data object
        edge_index: [2, E'] edge index (possibly pruned)
        config: full config
        num_features: input feature dim
        num_classes: number of classes
        device: torch device
        seed: random seed

    Returns:
        dict with val_acc, test_acc, best_epoch, runtime
    """
    set_seed(seed)
    logger = get_logger("train_downstream")

    ds_cfg = config.get("downstream_model", {})
    train_cfg = config["training"]

    model = build_model(
        name=model_name,
        in_dim=num_features,
        hidden_dim=ds_cfg.get("hidden_dim", 64),
        out_dim=num_classes,
        num_layers=ds_cfg.get("num_layers", 2),
        dropout=ds_cfg.get("dropout", 0.5),
    ).to(device)

    x = data.x.to(device)
    edge_index = edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 0.01),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
    )

    patience = train_cfg.get("patience", 100)
    early_stopping = EarlyStopping(patience=patience)
    epochs = train_cfg.get("epochs", 300)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(x, edge_index)
        loss = supervised_loss(logits, y, train_mask)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits_eval = model(x, edge_index)
            val_metrics = evaluate(logits_eval, y, val_mask)

        if early_stopping.step(val_metrics["accuracy"], model, epoch):
            break

    runtime = time.time() - start_time

    # Restore best and evaluate on test
    early_stopping.load_best_model(model)
    model.eval()
    with torch.no_grad():
        logits_test = model(x, edge_index)
        test_metrics = evaluate(logits_test, y, test_mask)
        val_metrics_final = evaluate(logits_test, y, val_mask)

    result = {
        "val_acc": val_metrics_final["accuracy"],
        "test_acc": test_metrics["accuracy"],
        "test_f1": test_metrics["macro_f1"],
        "best_epoch": early_stopping.best_epoch,
        "runtime": runtime,
    }

    logger.info(
        f"{model_name} downstream: val_acc={result['val_acc']:.4f}, "
        f"test_acc={result['test_acc']:.4f}, best_epoch={result['best_epoch']}, "
        f"runtime={runtime:.1f}s"
    )

    return result
