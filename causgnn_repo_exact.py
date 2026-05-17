import torch
import torch.nn.functional as F

class CausGNN_RepoExact:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def explain(self, data, target_class):
        """
        CausGNN replica from original paper
        """
        x = data.x
        edge_index = data.edge_index
        batch = torch.zeros(x.shape[0], dtype=torch.long).to(self.device)

        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        node_scores = torch.zeros(num_nodes, dtype=torch.float32, device=self.device)
        edge_scores = torch.zeros(num_edges, dtype=torch.float32, device=self.device)

        # Original Prob
        with torch.no_grad():
            try:
                orig_out = self.model(x, edge_index, batch=batch)
            except TypeError:
                orig_out = self.model(x, edge_index, None, batch)
            orig_prob = F.softmax(orig_out, dim=-1)[0, target_class].item()

        # Nodes Causality
        for i in range(num_nodes):
            x_perturbed = x.clone()
            x_perturbed[i] = 0.0 

            with torch.no_grad():
                try:
                    pert_out = self.model(x_perturbed, edge_index, batch=batch)
                except TypeError:
                    pert_out = self.model(x_perturbed, edge_index, None, batch)
                pert_prob = F.softmax(pert_out, dim=-1)[0, target_class].item()

            prob_drop = orig_prob - pert_prob
            node_scores[i] = max(0.0, prob_drop)

        # Edges Causality
        for j in range(num_edges):
            # mask (except edge j)
            edge_mask_keep = torch.ones(num_edges, dtype=torch.bool, device=self.device)
            edge_mask_keep[j] = False
            modified_edge_index = edge_index[:, edge_mask_keep]

            with torch.no_grad():
                try:
                    pert_out = self.model(x, modified_edge_index, batch=batch)
                except TypeError:
                    # edge_attr
                    mod_attr = data.edge_attr[edge_mask_keep] if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
                    pert_out = self.model(x, modified_edge_index, mod_attr, batch)
                pert_prob = F.softmax(pert_out, dim=-1)[0, target_class].item()

            prob_drop = orig_prob - pert_prob
            edge_scores[j] = max(0.0, prob_drop)

        # Min-Max Norm
        if node_scores.max() > 0:
            node_scores = (node_scores - node_scores.min()) / (node_scores.max() - node_scores.min())
        if edge_scores.max() > 0:
            edge_scores = (edge_scores - edge_scores.min()) / (edge_scores.max() - edge_scores.min())

        return node_scores.cpu(), edge_scores.cpu()
