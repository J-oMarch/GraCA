import torch
from src.graca.gradient_collector import collect_hidden_gradients
from src.graca.edge_scoring import compute_edge_scores, average_undirected_scores
from src.graca.pruning import prune_graph
from src.graca.save_graph import save_sanitized_graph
from src.training.losses import supervised_loss
from src.utils.logger import get_logger


def collect_oracle_gradients(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    y: torch.Tensor,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
) -> dict:
    """ORACLE ONLY: collect gradients using full labels for diagnostic upper-bound.

    Do not include in main semi-supervised results.
    """
    # ORACLE ONLY: uses full labels for diagnostic upper-bound.
    # Do not include in main semi-supervised results.

    model.train()
    model.zero_grad(set_to_none=True)

    logits, hidden_list = model(
        x, edge_index, return_hidden=True, retain_hidden_grad=True
    )

    hidden = hidden_list[-1]

    # Oracle uses ALL labels (train + val + test)
    all_mask = train_mask | test_mask
    loss = supervised_loss(logits, y, all_mask)

    loss.backward()

    grad = hidden.grad.detach().clone()
    hidden_detached = hidden.detach().clone()

    model.zero_grad(set_to_none=True)

    return {
        "hidden": hidden_detached,
        "grad": grad,
        "logits": logits.detach(),
        "loss_score": loss.item(),
    }


def run_oracle_graca(config: dict, model, data, device) -> dict:
    """Run Oracle GraCA pipeline.

    Requires: config['experiment']['oracle_only'] == True
    """
    assert config["experiment"]["oracle_only"] is True, (
        "Oracle mode requires oracle_only=true"
    )
    assert config["experiment"]["method"] == "oracle", (
        "Oracle mode requires method=oracle"
    )

    logger = get_logger("oracle")
    logger.info("Running Oracle GraCA (full-label diagnostic)")

    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    y = data.y.to(device)
    train_mask = data.train_mask.to(device)
    test_mask = data.test_mask.to(device)

    # Collect gradients with full labels
    grad_result = collect_oracle_gradients(
        model, x, edge_index, y, train_mask, test_mask
    )

    grad = grad_result["grad"]
    num_nodes = x.shape[0]

    # Oracle: all nodes have reliability 1
    rho_score = torch.ones(num_nodes, device=device)

    # Compute edge scores
    scoring_cfg = config.get("scoring", {})
    edge_scores = compute_edge_scores(
        grad=grad,
        edge_index=edge_index,
        rho_score=rho_score,
        num_nodes=num_nodes,
        eta=scoring_cfg.get("eta", 1.0),
        epsilon_rho=config.get("pseudo", {}).get("epsilon_rho", 0.05),
    )

    P = edge_scores["P"]

    # Average undirected scores
    if config.get("dataset", {}).get("undirected", True):
        P = average_undirected_scores(edge_index, P)

    # Prune
    pruning_cfg = config.get("pruning", {})
    pruned_edge_index, prune_mask, graph_stats = prune_graph(
        edge_index=edge_index,
        risk_score=P,
        num_nodes=num_nodes,
        beta=pruning_cfg.get("beta", 0.2),
        min_degree=pruning_cfg.get("min_degree", 1),
        lambda_theta=pruning_cfg.get("lambda_theta", 0.0),
        undirected=config.get("dataset", {}).get("undirected", True),
        protect_self_loops=pruning_cfg.get("protect_self_loops", True),
    )

    logger.info(f"Oracle pruning stats: {graph_stats}")

    # Save
    save_dir = config.get("logging", {}).get("graph_dir", "sanitized_graphs/oracle/")
    seed = config.get("experiment", {}).get("seeds", [42])[0]
    ds_name = config["dataset"]["name"]
    filename = f"{ds_name}_oracle_seed{seed}"
    graph_path = save_sanitized_graph(
        pruned_edge_index, prune_mask, graph_stats, save_dir, filename
    )

    return {
        "pruned_edge_index": pruned_edge_index,
        "prune_mask": prune_mask,
        "graph_stats": graph_stats,
        "graph_path": graph_path,
        "edge_scores": {k: v.cpu() for k, v in edge_scores.items()},
    }
