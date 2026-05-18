import torch

def keep_test(important_nodes_local: list, model, sub_x, sub_edge_index, target_sub_idx, device) -> float:
    imp_set = set(important_nodes_local)
    mask    = torch.tensor(
        [s in imp_set and d in imp_set
         for s, d in sub_edge_index.t().tolist()],
        dtype=torch.bool, device=device
    )
    ei_keep = sub_edge_index[:, mask]
    with torch.no_grad():
        out = model(sub_x, ei_keep)
    return float(out.exp()[target_sub_idx, 1])

def deletion_test(important_nodes_local: list, model, sub_x, sub_edge_index, target_sub_idx, device) -> float:
    to_remove = set(important_nodes_local) - {target_sub_idx}
    mask      = torch.tensor(
        [s not in to_remove and d not in to_remove
         for s, d in sub_edge_index.t().tolist()],
        dtype=torch.bool, device=device
    )
    ei_del = sub_edge_index[:, mask]
    with torch.no_grad():
        out = model(sub_x, ei_del)
    return float(out.exp()[target_sub_idx, 1])

def compute_fidelity(explanation, model, sub_x, sub_edge_index, target_sub_idx, threshold, device=None):
    THRESHOLD_IG = threshold
    node_mask_ig = explanation.node_mask.mean(dim=-1).cpu()
    if device is None:
        device = sub_x.device

    important_nodes_ig = [
        int(i) for i in range(sub_x.size(0))
        if float(node_mask_ig[i]) >= THRESHOLD_IG
    ]
    if target_sub_idx not in important_nodes_ig:
        important_nodes_ig.append(target_sub_idx)

    with torch.no_grad():
        base_score_ig = float(model(sub_x, sub_edge_index).exp()[target_sub_idx, 1])

    keep_score_ig     = keep_test(important_nodes_ig, model, sub_x, sub_edge_index, target_sub_idx, device)
    deletion_score_ig = deletion_test(important_nodes_ig, model, sub_x, sub_edge_index, target_sub_idx, device)

    return base_score_ig, keep_score_ig, deletion_score_ig, important_nodes_ig

def sparsity(explanation, sub_x):
    node_mask = explanation.node_mask.mean(dim=-1).cpu()
    return float((node_mask < 0.5).sum()) / sub_x.size(0)

def compute_base_score(model, x, edge_index, target_idx, target_class=1):
    model.eval()

    with torch.no_grad():
        out = model(x, edge_index)
        prob = float(out.exp()[target_idx, target_class].item())

    return prob

def keep_top_edges_test(
    edge_scores,
    model,
    sub_x,
    sub_edge_index,
    target_sub_idx,
    top_k=10,
    target_class=1,
):
    model.eval()

    edge_scores = edge_scores.detach().cpu()
    top_k = min(top_k, edge_scores.numel())

    top_edge_indices = torch.topk(edge_scores, k=top_k).indices.to(sub_edge_index.device)

    keep_mask = torch.zeros(
        sub_edge_index.size(1),
        dtype=torch.bool,
        device=sub_edge_index.device,
    )
    keep_mask[top_edge_indices] = True

    kept_edge_index = sub_edge_index[:, keep_mask]

    with torch.no_grad():
        out = model(sub_x, kept_edge_index)
        score = float(out.exp()[target_sub_idx, target_class].item())

    return score, top_edge_indices.detach().cpu()

def delete_top_edges_test(
    edge_scores,
    model,
    sub_x,
    sub_edge_index,
    target_sub_idx,
    top_k=10,
    target_class=1,
):
    model.eval()

    edge_scores = edge_scores.detach().cpu()
    top_k = min(top_k, edge_scores.numel())

    top_edge_indices = torch.topk(edge_scores, k=top_k).indices.to(sub_edge_index.device)

    delete_mask = torch.ones(
        sub_edge_index.size(1),
        dtype=torch.bool,
        device=sub_edge_index.device,
    )
    delete_mask[top_edge_indices] = False

    deleted_edge_index = sub_edge_index[:, delete_mask]

    with torch.no_grad():
        out = model(sub_x, deleted_edge_index)
        score = float(out.exp()[target_sub_idx, target_class].item())

    return score

def count_edges_to_target(edge_indices, sub_edge_index, target_sub_idx):
    count = 0

    for edge_idx in edge_indices:
        edge_idx = int(edge_idx.item())
        target_local = int(sub_edge_index[1, edge_idx].item())

        if target_local == target_sub_idx:
            count += 1

    return count

def normalize_scores(scores):
    scores = scores.detach().cpu().float()

    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)