import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, GraphNorm

class ManualGINEConv(MessagePassing):
    def __init__(self, mlp, eps=0., train_eps=True):
        super().__init__(aggr='add') 
        self.nn = mlp
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        x_r = x * (1 + self.eps)
        return self.nn(out + x_r)

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

class DockingGNN_Robust(torch.nn.Module):
    def __init__(self, node_dim=22, edge_dim=4, hidden_dim=64, num_layers=3, dropout=0.0): 
        super().__init__()
        
        self.input_norm = nn.LayerNorm(node_dim)
        
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            )
            self.convs.append(ManualGINEConv(mlp, train_eps=True))
            self.norms.append(GraphNorm(hidden_dim))
            
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim) 
        self.lin2 = nn.Linear(hidden_dim, 1)
        
        self.dropout_rate = dropout

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x = self.input_norm(x)
        
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x, batch) 
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1) 
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        
        return x