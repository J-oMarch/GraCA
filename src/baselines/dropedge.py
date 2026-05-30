import torch
from src.models.model_factory import build_model
from src.training.losses import supervised_loss
from src.training.evaluator import evaluate
from src.training.early_stopping import EarlyStopping
from src.utils.seed import set_seed
from src.utils.logger import get_logger
import time


def run_dropedge(data, config, num_features, num_classes, device, seed=42):
    """DropEdge baseline: randomly drop edges during training."""
    set_seed(seed)
    logger = get_logger("dropedge")

    drop_rate = config.get("baselines", {}).get("dropedge_rate", 0.2)
    downstream_names = config.get("downstream_model", {}).get("names", ["GCN"])
    ds_cfg = config.get("downstream_model", {})
    train_cfg = config["training"]

    results = {}

    for model_name in downstream_names:
        model = build_model(
            name=model_name,
            in_dim=num_features,
            hidden_dim=ds_cfg.get("hidden_dim", 64),
            out_dim=num_classes,
            num_layers=ds_cfg.get("num_layers", 2),
            dropout=ds_cfg.get("dropout", 0.5),
        ).to(device)

        x = data.x.to(device)
        edge_index = data.edge_index.to(device)
        y = data.y.to(device)
        train_mask = data.train_mask.to(device)
        val_mask = data.val_mask.to(device)
        test_mask = data.test_mask.to(device)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_cfg.get("lr", 0.01),
            weight_decay=train_cfg.get("weight_decay", 0.0005),
        )

        early_stopping = EarlyStopping(patience=train_cfg.get("patience", 100))
        epochs = train_cfg.get("epochs", 300)

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()

            # Random edge dropout
            E = edge_index.shape[1]
            keep_mask = torch.rand(E, device=device) > drop_rate
            # Always keep self-loops
            self_loop_mask = edge_index[0] == edge_index[1]
            keep_mask = keep_mask | self_loop_mask
            dropped_edge_index = edge_index[:, keep_mask]

            logits = model(x, dropped_edge_index)
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
        early_stopping.load_best_model(model)
        model.eval()
        with torch.no_grad():
            logits_test = model(x, edge_index)
            test_metrics = evaluate(logits_test, y, test_mask)

        results[model_name] = {
            "val_acc": evaluate(logits_test, y, val_mask)["accuracy"],
            "test_acc": test_metrics["accuracy"],
            "test_f1": test_metrics["macro_f1"],
            "best_epoch": early_stopping.best_epoch,
            "runtime": runtime,
        }

        logger.info(
            f"DropEdge {model_name}: test_acc={results[model_name]['test_acc']:.4f}"
        )

    return results
