import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    PNAConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)
from torch_geometric.utils import degree


class PolyatomicNet(nn.Module):
    def __init__(
        self,
        node_feat_dim,
        edge_feat_dim,
        graph_feat_dim,
        deg,
        hidden_dim=128,
        num_layers=5,
        dropout=0.1,
    ):
        super().__init__()

        self.node_emb = nn.Linear(node_feat_dim, hidden_dim)
        self.deg = deg
        self.virtualnode_emb = nn.Embedding(1, hidden_dim)
        self.vn_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # For graph-level feature projection
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # PNAConv requires degree preprocessing
        self.deg_emb = nn.Embedding(20, hidden_dim)  # cap degree buckets

        aggregators = ["mean", "min", "max", "std"]
        scalers = ["identity", "amplification", "attenuation"]

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(num_layers):
            conv = PNAConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                aggregators=aggregators,
                scalers=scalers,
                edge_dim=edge_feat_dim,
                towers=4,
                pre_layers=1,
                post_layers=1,
                divide_input=True,
                deg=deg,
            )
            self.convs.append(conv)
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Final readout
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        deg = degree(edge_index[0], x.size(0), dtype=torch.long).clamp(max=19)
        h = self.node_emb(x) + self.deg_emb(deg)

        vn = self.virtualnode_emb(
            torch.zeros(batch.max().item() + 1, dtype=torch.long, device=x.device)
        )

        for conv, bn in zip(self.convs, self.bns):
            h = h + vn[batch]
            h = conv(h, edge_index, edge_attr)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
            vn = vn + self.vn_mlp(global_mean_pool(h, batch))

        mean_pool = global_mean_pool(h, batch)
        max_pool = global_max_pool(h, batch)
        add_pool = global_add_pool(h, batch)

        # graph_feats shape: [batch_size, graph_feat_dim]
        if hasattr(data, "graph_feats") and isinstance(
            data, torch_geometric.data.Batch  # type: ignore
        ):
            g_feats = torch.stack(
                [g.graph_feats for g in data.to_data_list()], dim=0
            ).to(x.device)
        else:
            g_feats = data.graph_feats.unsqueeze(0).to(
                x.device
            )  # handles single graph batch
        g_proj = self.graph_proj(g_feats)

        final_input = torch.cat([mean_pool, max_pool, g_proj], dim=1)
        return self.readout(final_input).view(-1)
