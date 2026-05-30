import torch
import copy
from src.models.model_factory import build_model
from src.graca.ema_teacher import EMATeacher
from src.graca.pseudo_label import compute_soft_pseudo_labels, compute_pseudo_coverage
from src.training.losses import supervised_loss, soft_pseudo_loss
from src.training.evaluator import evaluate
from src.training.early_stopping import EarlyStopping
from src.data.leakage_check import ensure_no_test_label_leakage
from src.utils.seed import set_seed
from src.utils.logger import get_logger


def train_proxy(config: dict, data, num_features: int, num_classes: int, device, seed: int = 42):
    """Train ProxyGNN with EMA teacher, soft pseudo labels, and optional consistency loss.

    Returns:
        model: trained ProxyGNN
        teacher: EMA teacher
        training_log: dict with training statistics
    """
    set_seed(seed)
    logger = get_logger("train_proxy")

    # Config
    proxy_cfg = config["proxy_model"]
    train_cfg = config["training"]
    teacher_cfg = config.get("teacher", {})
    pseudo_cfg = config.get("pseudo", {})
    scoring_cfg = config.get("scoring", {})
    consistency_cfg = config.get("consistency", {})

    # Build model
    model = build_model(
        name=proxy_cfg["name"],
        in_dim=num_features,
        hidden_dim=proxy_cfg["hidden_dim"],
        out_dim=num_classes,
        num_layers=proxy_cfg.get("num_layers", 2),
        dropout=proxy_cfg.get("dropout", 0.5),
    ).to(device)

    # EMA Teacher - will be initialized after warmup epochs
    use_ema = teacher_cfg.get("use_ema", True)
    teacher = None
    ema_warmup = teacher_cfg.get("warmup_epochs", 50)
    ema_initialized = False

    # Consistency loss
    use_consistency = consistency_cfg.get("use_consistency", False)
    lambda_c = consistency_cfg.get("lambda_c", 0.1)

    # Multi-checkpoint saving
    use_multi_checkpoint = scoring_cfg.get("use_multi_checkpoint", False)
    checkpoint_epochs = scoring_cfg.get("checkpoints", [100, 150, 200, 250, 300])
    saved_checkpoints = []

    # Data
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    val_mask = data.val_mask.to(device)
    test_mask = data.test_mask.to(device)
    unlabeled_mask = ~train_mask

    # Leakage check
    ensure_no_test_label_leakage(config, train_mask, test_mask, train_mask, mode="practical")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 0.01),
        weight_decay=train_cfg.get("weight_decay", 0.0005),
    )

    # Early stopping
    patience = train_cfg.get("patience", 100)
    early_stopping = EarlyStopping(patience=patience)

    # Pseudo label config
    tau = pseudo_cfg.get("tau", 0.8)
    alpha = pseudo_cfg.get("alpha", 1.0)
    lambda_p = pseudo_cfg.get("lambda_p", 1.0)
    eps_rho = pseudo_cfg.get("epsilon_rho", 0.05)

    epochs = train_cfg.get("epochs", 300)

    training_log = {
        "loss_sup": [],
        "loss_soft": [],
        "loss_cons": [],
        "pseudo_coverage": [],
        "mean_confidence": [],
        "val_acc": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(x, edge_index)

        # Supervised loss on train nodes
        loss_sup = supervised_loss(logits, y, train_mask)

        # Initialize EMA teacher after warmup
        if use_ema and not ema_initialized and epoch == ema_warmup:
            teacher = EMATeacher(model, decay=teacher_cfg.get("ema_decay", 0.99))
            teacher = teacher.to(device)
            ema_initialized = True
            logger.info(f"EMA teacher initialized at epoch {epoch}")

        # Soft pseudo loss
        loss_soft_val = torch.tensor(0.0, device=device)
        loss_cons_val = torch.tensor(0.0, device=device)
        pseudo_cov = 0.0
        mean_conf = 0.0

        if use_ema and ema_initialized:
            teacher_probs = teacher.predict(x, edge_index)
            q, confidence, entropy, rho_train = compute_soft_pseudo_labels(
                teacher_probs, train_mask, unlabeled_mask, tau, alpha, eps_rho
            )
            loss_soft_val, _ = soft_pseudo_loss(
                logits, teacher_probs, rho_train, unlabeled_mask
            )
            stats = compute_pseudo_coverage(unlabeled_mask, confidence, tau)
            pseudo_cov = stats["coverage"]
            mean_conf = stats["mean_confidence"]

            # Consistency loss
            if use_consistency:
                from src.graca.consistency_loss import compute_consistency_loss
                loss_cons_val, _, _ = compute_consistency_loss(
                    model, x, edge_index, rho_train, lambda_c,
                    consistency_cfg.get("edge_dropout", 0.3),
                    consistency_cfg.get("feature_mask", 0.3),
                )

        loss = loss_sup + lambda_p * loss_soft_val + loss_cons_val
        loss.backward()
        optimizer.step()

        # Update teacher
        if use_ema and ema_initialized:
            teacher.update(model)

        # Save checkpoint
        if use_multi_checkpoint and epoch in checkpoint_epochs:
            saved_checkpoints.append(copy.deepcopy(model.state_dict()))
            logger.info(f"Saved checkpoint at epoch {epoch}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            logits_eval = model(x, edge_index)
            val_metrics = evaluate(logits_eval, y, val_mask)

        # Logging
        training_log["loss_sup"].append(loss_sup.item())
        training_log["loss_soft"].append(loss_soft_val.item())
        training_log["loss_cons"].append(loss_cons_val.item())
        training_log["pseudo_coverage"].append(pseudo_cov)
        training_log["mean_confidence"].append(mean_conf)
        training_log["val_acc"].append(val_metrics["accuracy"])

        if epoch % 50 == 0:
            logger.info(
                f"Epoch {epoch}: loss_sup={loss_sup.item():.4f}, "
                f"loss_soft={loss_soft_val.item():.4f}, "
                f"loss_cons={loss_cons_val.item():.4f}, "
                f"pseudo_cov={pseudo_cov:.3f}, "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

        # Early stopping
        if early_stopping.step(val_metrics["accuracy"], model, epoch):
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Restore best model
    early_stopping.load_best_model(model)
    logger.info(f"Best epoch: {early_stopping.best_epoch}, best val_acc: {early_stopping.best_score:.4f}")

    # Save best checkpoint if not already saved
    if use_multi_checkpoint and early_stopping.best_epoch not in checkpoint_epochs:
        saved_checkpoints.append(copy.deepcopy(model.state_dict()))

    return model, teacher, training_log, saved_checkpoints
