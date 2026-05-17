# This file defines a compact ORExplainer-style method that learns an edge mask using factual and counterfactual objectives.

import torch
import torch.nn as nn
from torch_geometric.explain.algorithm.utils import set_masks, clear_masks

def run_orexplainer(
    model,
    sub_x,
    sub_edge_index,
    target_sub_idx,
    target_class=1,
    epochs=200,
    lr=0.01,
    lambda_cf=0.5,
    lambda_sparsity=0.05,
    lambda_entropy=0.02,
    device="cuda",
    verbose=False,
):
    model.eval()

    sub_x = sub_x.to(device)
    sub_edge_index = sub_edge_index.to(device)

    edge_logits = nn.Parameter(torch.randn(sub_edge_index.size(1), device=device) * 0.1)
    optimizer = torch.optim.Adam([edge_logits], lr=lr)

    with torch.no_grad():
        base_out = model(sub_x, sub_edge_index)
        base_score = float(base_out.exp()[target_sub_idx, target_class].item())

    for epoch in range(epochs):
        optimizer.zero_grad()

        edge_mask = torch.sigmoid(edge_logits)

        # Factual branch: keep the selected edges
        set_masks(model, edge_mask, sub_edge_index, apply_sigmoid=False)
        out_factual = model(sub_x, sub_edge_index)
        clear_masks(model)

        prob_factual = out_factual.exp()[target_sub_idx, target_class]
        loss_factual = -torch.log(prob_factual + 1e-8)

        # Counterfactual branch: remove the selected edges
        cf_mask = 1.0 - edge_mask

        set_masks(model, cf_mask, sub_edge_index, apply_sigmoid=False)
        out_counterfactual = model(sub_x, sub_edge_index)
        clear_masks(model)

        prob_counterfactual = out_counterfactual.exp()[target_sub_idx, target_class]
        loss_counterfactual = prob_counterfactual

        loss_sparse = edge_mask.mean()

        loss_entropy = -(
            edge_mask * torch.log(edge_mask + 1e-8)
            + (1 - edge_mask) * torch.log(1 - edge_mask + 1e-8)
        ).mean()

        loss = (
            loss_factual
            + lambda_cf * loss_counterfactual
            + lambda_sparsity * loss_sparse
            + lambda_entropy * loss_entropy
        )

        loss.backward()
        optimizer.step()

        if verbose and epoch % 50 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Factual P(fraud): {prob_factual.item():.4f} | "
                f"Counterfactual P(fraud): {prob_counterfactual.item():.4f}"
            )

    clear_masks(model)

    with torch.no_grad():
        edge_scores = torch.sigmoid(edge_logits).detach().cpu()
        edge_scores = (edge_scores - edge_scores.min()) / (
            edge_scores.max() - edge_scores.min() + 1e-8
        )

    return edge_scores, base_score