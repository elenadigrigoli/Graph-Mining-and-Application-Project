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