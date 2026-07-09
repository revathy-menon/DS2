#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.nn import MessagePassing, global_mean_pool, global_max_pool, GraphNorm
from torch_geometric.utils import softmax
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dropout_edge
from datetime import datetime
from tqdm import tqdm


# --- CONFIGURATION ---
CONFIG = {
    'graphs_dir': '.', # <--- UPDATE THIS
    'model_path': 'models/gnn_hard_mining_with_edge_features_v2/gnn_with_edge_model_v10.pt',
    'report_file': f'output_{datetime.now().strftime("%Y%m%d")}.txt', # <--- UPDATE THIS
}

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
        
        # Input normalization to prevent exploding raw node features
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

def run_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DockingGNN_Robust(node_dim=22, edge_dim=4, hidden_dim=64).to(device)
    
    try:
        checkpoint = torch.load(CONFIG['model_path'], map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print("PASS: Model loaded successfully! (Strict Match)")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    pt_files = [f for f in os.listdir(CONFIG['graphs_dir']) if f.endswith('.pt')]
    if not pt_files:
        print("No graph files found.")
        sys.exit(1)

    print(f"Auditing {len(pt_files)} targets...")
    results = []

    for pt_file in tqdm(pt_files):
        target_id = pt_file.replace(".pt", "")
        file_path = os.path.join(CONFIG['graphs_dir'], pt_file)
        
        try:
            graphs = torch.load(file_path)
            
            # Filter Native
            predictor_graphs = [g for g in graphs if g.id != 'NATIVE']
            if not predictor_graphs: continue
            
            # Inference
            loader = DataLoader(predictor_graphs, batch_size=64, shuffle=False)
            all_scores = []
            all_dockq = []

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    
                    # Clamp Inverse Distance to a safe max (1.0 is close to BM5's 0.84)
                    batch.edge_attr[:, 0] = torch.clamp(batch.edge_attr[:, 0], max=1.0)
                    
                    # Clamp Coulombic electrostatics to safe bounds
                    batch.edge_attr[:, 3] = torch.clamp(batch.edge_attr[:, 3], min=-2.0, max=2.0)

                    all_dockq.extend(batch.y.cpu().numpy().flatten())
                    scores = model(batch).cpu().numpy().flatten()
                    all_scores.extend(scores)
            
            # Metrics
            df = pd.DataFrame({'dockq': all_dockq, 'score': all_scores})
            df = df.sort_values('score', ascending=False)
            
            top1_dq = df.iloc[0]['dockq']
            max_dq = df['dockq'].max()
            
            results.append({
                'Target': target_id,
                'Max_DockQ': max_dq,
                'Top1_DockQ': top1_dq,
                'Success@1': 1 if top1_dq >= 0.23 else 0,
                'Success@10': 1 if (df.head(10)['dockq'] >= 0.23).any() else 0
            })
            
        except Exception:
            continue

    if results:
        df_res = pd.DataFrame(results)
        solvable = df_res[df_res['Max_DockQ'] >= 0.23]
        
        print("\n" + "="*60)
        print("FINAL AUDIT RESULTS")
        print("="*60)
        print(f"Total Targets: {len(df_res)}")
        print(f"Solvable Targets: {len(solvable)}")
        print(f"Success Rate @ 1:  {solvable['Success@1'].mean()*100:.1f}%")
        print(f"Success Rate @ 10: {solvable['Success@10'].mean()*100:.1f}%")
        
        with open(CONFIG['report_file'], 'w') as f:
            f.write(df_res.to_string())
        print(f"Report saved to {CONFIG['report_file']}")

if __name__ == "__main__":
    run_audit()
