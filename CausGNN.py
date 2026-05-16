import torch
import torch.nn as nn
import pandas as pd

class CausGNNExplainer(nn.Module):
    """CausGNN-style node mask explainer for local structural explanations."""

    def __init__(self, num_nodes):
        super().__init__()
        self.mask_logits = nn.Parameter(torch.randn(num_nodes))

    def forward(self):
        return torch.sigmoid(self.mask_logits)


def run_causgnn_explainer(
    model,
    sub_x,
    sub_edge_index,
    target_sub_idx,
    target_class=1,
    lambda_cf=0.5,
    lambda_sparsity=0.08,
    lambda_entropy=0.02,
    epochs=300,
    lr=0.01,
    device="cuda",
    verbose=True,
):
    """Learn a node-level structural mask using a factual/counterfactual objective."""
    model.eval()

    sub_x = sub_x.to(device)
    sub_edge_index = sub_edge_index.to(device)

    num_nodes = sub_x.size(0)

    explainer = CausGNNExplainer(num_nodes).to(device)
    optimizer_caus = torch.optim.Adam(explainer.parameters(), lr=lr)

    caus_history = []

    # Baseline prediction before learning the explanation mask
    with torch.no_grad():
        base_out = model(sub_x, sub_edge_index)
        base_probs = base_out.exp()[target_sub_idx]
        base_score = float(base_probs[target_class].item())
        base_pred = int(base_probs.argmax().item())

    for epoch in range(1, epochs + 1):
        optimizer_caus.zero_grad()

        node_mask = explainer()

        # The target node must remain active
        mask_for_use = node_mask.clone()
        mask_for_use[target_sub_idx] = 1.0

        # Factual branch: keep important nodes active
        x_factual = sub_x * mask_for_use.unsqueeze(1)
        out_factual = model(x_factual, sub_edge_index)
        prob_factual = out_factual.exp()[target_sub_idx, target_class]

        loss_factual = -torch.log(prob_factual + 1e-8)

        # Counterfactual branch: remove important nodes
        x_counterfactual = sub_x * (1.0 - mask_for_use).unsqueeze(1)
        out_counterfactual = model(x_counterfactual, sub_edge_index)
        prob_counterfactual = out_counterfactual.exp()[target_sub_idx, target_class]

        loss_counterfactual = prob_counterfactual

        # Regularization terms
        loss_sparsity = node_mask.mean()

        loss_entropy = -(
            node_mask * torch.log(node_mask + 1e-8)
            + (1.0 - node_mask) * torch.log(1.0 - node_mask + 1e-8)
        ).mean()

        loss = (
            loss_factual
            + lambda_cf * loss_counterfactual
            + lambda_sparsity * loss_sparsity
            + lambda_entropy * loss_entropy
        )

        loss.backward()
        optimizer_caus.step()

        caus_history.append(
            {
                "epoch": epoch,
                "loss": float(loss.item()),
                "factual_p_fraud": float(prob_factual.item()),
                "counterfactual_p_fraud": float(prob_counterfactual.item()),
                "mask_mean": float(node_mask.mean().item()),
            }
        )

        if verbose and (epoch == 1 or epoch % 50 == 0):
            print(
                f"Epoch {epoch:03d} | "
                f"Loss: {loss.item():.4f} | "
                f"Factual P(fraud): {prob_factual.item():.4f} | "
                f"Counterfactual P(fraud): {prob_counterfactual.item():.4f} | "
                f"Mask mean: {node_mask.mean().item():.4f}"
            )

    with torch.no_grad():
        final_mask = explainer().detach().cpu()

        # Force the target node to have maximum relevance
        final_mask[target_sub_idx] = 1.0

        # Normalize node scores to [0, 1]
        final_mask = (
            (final_mask - final_mask.min())
            / (final_mask.max() - final_mask.min() + 1e-8)
        )

    caus_history_df = pd.DataFrame(caus_history)

    return final_mask, base_score, base_pred, caus_history_df