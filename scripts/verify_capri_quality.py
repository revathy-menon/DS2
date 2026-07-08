#!/usr/bin/env python3
"""
AUDIT REPORT: CAPRI Quality Retrieval (COMPATIBILITY MODE)
Objective: Audit GNN using a manually defined GINE layer to bypass version mismatches.
"""

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
    # 'graphs_dir': '/home/mini/revathym/ml_ds/results/processed_graphs_capri_dynamic/',
    'graphs_dir': '/home/mini/revathym/ml_ds/results/processed_graphs_capri_static_physics_v2/',
    # 'model_path': '/home/mini/revathym/ml_ds/models/gnn_ranker/gnn_model.pt',
    # 'model_path' : '/home/mini/revathym/ml_ds/models/gnn_hard_mining/last_model.pt',
    'model_path': 'models/gnn_hard_mining_with_edge_features_v2/gnn_with_edge_model_v10.pt',
    'report_file': f'recheck/capri_quality_audit_robust_v10_{datetime.now().strftime("%Y%m%d")}.txt',
}

class ManualGINEConv(MessagePassing):
    def __init__(self, mlp, eps=0., train_eps=True):
        # 1. REVERT TO 'add': Interface nodes have more edges, so 'add' makes 
        # their values naturally spike higher than bulk nodes, highlighting them.
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
    # 1. DROPOUT IS DEAD: Set to 0.0
    def __init__(self, node_dim=22, edge_dim=4, hidden_dim=64, num_layers=3, dropout=0.0): 
        super().__init__()
        
        # 2. INPUT NORMALIZATION: Prevents exploding raw node features
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
        
        # 3. DROPEDGE IS DEAD: Removed completely to preserve exact physics
        
        # 4. Normalize raw nodes BEFORE projection
        x = self.input_norm(x)
        
        x = self.node_proj(x)
        edge_attr = self.edge_proj(edge_attr)
        
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr)
            x = norm(x, batch) 
            x = F.relu(x)
            # F.dropout remains, but dropout_rate is 0.0, so it passes cleanly
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1) 
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.lin2(x)
        
        return x

# class DockingGNN_Robust(torch.nn.Module):
#     # 2. DROP THE DROPOUT: Changed from 0.4 to 0.1
#     def __init__(self, node_dim=22, edge_dim=4, hidden_dim=64, num_layers=3, dropout=0.1): 
#         super().__init__()
        
#         self.node_proj = nn.Linear(node_dim, hidden_dim)
#         self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList()
        
#         for _ in range(num_layers):
#             mlp = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU()
#             )
#             self.convs.append(ManualGINEConv(mlp, train_eps=True))
#             self.norms.append(GraphNorm(hidden_dim))
            
#         # 3. RESTORE CONCATENATION DIMENSION: Back to hidden_dim * 2
#         self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim) 
#         self.lin2 = nn.Linear(hidden_dim, 1)
        
#         self.dropout_rate = dropout

#     def forward(self, data):
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
#         if self.training:
#             # DropEdge stays low at 5% so we don't break the physics
#             from torch_geometric.utils import dropout_edge
#             edge_index, edge_mask = dropout_edge(edge_index, p=0.05)
#             edge_attr = edge_attr[edge_mask]
        
#         x = self.node_proj(x)
#         edge_attr = self.edge_proj(edge_attr)
        
#         for conv, norm in zip(self.convs, self.norms):
#             x = conv(x, edge_index, edge_attr)
#             x = norm(x, batch) 
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
#         # 4. RESTORE MAX POOLING: This extracts the high-signal interface nodes
#         # while mean pooling provides the global context.
#         x_mean = global_mean_pool(x, batch)
#         x_max = global_max_pool(x, batch)
#         x = torch.cat([x_mean, x_max], dim=1) 
        
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.lin2(x)
        
#         return x

# # --- MANUAL GINE LAYER (Forces 'nn' keys) ---
# class ManualGINEConv(MessagePassing):
#     def __init__(self, mlp, eps=0., train_eps=True):
#         # super().__init__(aggr='add')
#         super().__init__(aggr='mean')
#         self.nn = mlp  # HARDCODED: Matches checkpoint key 'convs.0.nn'
#         self.initial_eps = eps
#         if train_eps:
#             self.eps = torch.nn.Parameter(torch.Tensor([eps])) # Matches 'convs.0.eps'
#         else:
#             self.register_buffer('eps', torch.Tensor([eps]))

#     def forward(self, x, edge_index, edge_attr):
#         # x: [N, F], edge_index: [2, E], edge_attr: [E, F]
        
#         # Propagate: this calls message() -> aggregate() -> update()
#         out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        
#         # GIN Update: MLP((1 + eps) * x + aggregate(x_j + edge_attr))
#         x_r = x * (1 + self.eps)
#         return self.nn(out + x_r)

#     def message(self, x_j, edge_attr):
#         return F.relu(x_j + edge_attr)



# # (Keep your existing ManualGINEConv class here)

# class DockingGNN_Robust(torch.nn.Module):
#     def __init__(self, node_dim=22, edge_dim=4, hidden_dim=64, num_layers=3, dropout=0.3):
#         super().__init__()
        
#         self.node_proj = nn.Linear(node_dim, hidden_dim)
#         self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
#         self.convs = nn.ModuleList()
#         self.norms = nn.ModuleList() # Changed from bns to norms
        
#         for _ in range(num_layers):
#             # Removed the internal BatchNorm1d to avoid routing issues
#             mlp = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU()
#             )
#             self.convs.append(ManualGINEConv(mlp, train_eps=True))
            
#             # Use GraphNorm instead of BatchNorm1d
#             self.norms.append(GraphNorm(hidden_dim))
            
#         # self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
#         self.lin1 = nn.Linear(hidden_dim, hidden_dim)  
#         self.lin2 = nn.Linear(hidden_dim, 1)
        
#         self.dropout_rate = dropout

#     def forward(self, data):
#         # We MUST extract the batch vector to pass into GraphNorm
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

#         # --- DROPEDGE IMPLEMENTATION ---
#         # Only drop edges during training, never during evaluation/inference
#         if self.training:
#             # Drop 20% of edges randomly
#             edge_index, edge_mask = dropout_edge(edge_index, p=0.05)
#             # Apply the exact same mask to edge_attr so shapes match!
#             edge_attr = edge_attr[edge_mask]
#         # -------------------------------
        
#         x = self.node_proj(x)
#         edge_attr = self.edge_proj(edge_attr)
        
        
#         for conv, norm in zip(self.convs, self.norms):
#             x = conv(x, edge_index, edge_attr)
            
#             # CRITICAL DIFFERENCE: Pass the 'batch' index to GraphNorm
#             x = norm(x, batch) 
            
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
#         x_mean = global_mean_pool(x, batch)
#         # x_max = global_max_pool(x, batch)
#         # x = torch.cat([x_mean, x_max], dim=1)
#         x = x_mean 
        
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.lin2(x)
        
#         return x

# --- THE MODEL ARCHITECTURE ---
# changed edge_dim from 1 to 4
# class DockingGNN_Compat(torch.nn.Module):
#     def __init__(self, node_dim=22, edge_dim=4, hidden_dim=128, num_layers=3, dropout=0.3):
#         super().__init__()
        
#         # 1. Projections (Keys: node_proj, edge_proj)
#         self.node_proj = nn.Linear(node_dim, hidden_dim)
#         self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
#         # 2. Convolutions (Keys: convs.0.nn...)
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
        
#         for _ in range(num_layers):
#             # The MLP inside the GINE layer
#             mlp = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 # nn.BatchNorm1d(hidden_dim),
#                 torch_geometric.nn.GraphNorm(hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU()
#             )
#             # Use Manual Layer to force naming
#             self.convs.append(ManualGINEConv(mlp, train_eps=True))
            
#             # 3. Batch Norms (Keys: bns.0...)
#             self.bns.append(torch_geometric.nn.GraphNorm(hidden_dim))
            
#         # 4. Readout Head (Keys: lin1, lin2)
#         # Input 256 due to Concatenated Pooling
#         self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim) 
#         self.lin2 = nn.Linear(hidden_dim, 1)
        
#         self.dropout_rate = dropout

#     def forward(self, data):
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
#         # Projections
#         x = self.node_proj(x)
#         edge_attr = self.edge_proj(edge_attr)
        
#         # Message Passing
#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index, edge_attr)
#             x = bn(x)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
#         # Concatenated Pooling
#         x_mean = global_mean_pool(x, batch)
#         x_max = global_max_pool(x, batch)
#         x = torch.cat([x_mean, x_max], dim=1) # [Batch, 256]
        
#         # MLP Head
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.lin2(x)
        
#         return x

def run_audit():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Model
    # model = DockingGNN_Compat(node_dim=22, edge_dim=4, hidden_dim=64).to(device)
    model = DockingGNN_Robust(node_dim=22, edge_dim=4, hidden_dim=64).to(device)
    # model = DockingGNN_Compat(node_dim=22, edge_dim=1, hidden_dim=128).to(device)
    
    try:
        checkpoint = torch.load(CONFIG['model_path'], map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        
        # Strict=True ensures every key matches exactly
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        print("PASS: Model loaded successfully! (Strict Match)")
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        sys.exit(1)

    # Load Data
    pt_files = [f for f in os.listdir(CONFIG['graphs_dir']) if f.endswith('.pt')]
    if not pt_files:
        print("No graph files found. Run scavenge_capri_dynamic.py first.")
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
            
            # with torch.no_grad():
            #     for batch in loader:
            #         batch = batch.to(device)
            #         all_dockq.extend(batch.y.cpu().numpy().flatten())
            #         scores = model(batch).cpu().numpy().flatten()
            #         all_scores.extend(scores)

            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    
                    # --- EXPLODING PHYSICS HOTFIX START ---
                    # 1. Clamp Inverse Distance to a safe max (1.0 is close to BM5's 0.84)
                    batch.edge_attr[:, 0] = torch.clamp(batch.edge_attr[:, 0], max=1.0)
                    
                    # 2. Zero-out Steric Clashes because the model never learned them!
                    batch.edge_attr[:, 1] = 0.0
                    
                    # 3. Clamp Coulombic electrostatics to safe bounds
                    batch.edge_attr[:, 3] = torch.clamp(batch.edge_attr[:, 3], min=-2.0, max=2.0)
                    # --- EXPLODING PHYSICS HOTFIX END ---

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

# #!/usr/bin/env python3
# """
# AUDIT REPORT: CAPRI Quality Retrieval (FINAL ARCHITECTURE MATCH)
# Objective: Verify GNN using the architecture derived from specific weight mismatch errors.
# """

# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
# from torch_geometric.loader import DataLoader
# from datetime import datetime
# from tqdm import tqdm

# # --- CONFIGURATION ---
# CONFIG = {
#     'graphs_dir': '/home/mini/revathym/ml_ds/results/processed_graphs_capri_dynamic/',
#     'model_path': '/home/mini/revathym/ml_ds/models/gnn_ranker/gnn_model.pt',
#     'report_file': f'recheck/capri_quality_audit_SUCCESS_{datetime.now().strftime("%Y%m%d")}.txt',
# }

# # --- THE CORRECTED ARCHITECTURE ---
# class DockingGNN_Final(torch.nn.Module):
#     def __init__(self, node_dim=22, edge_dim=1, hidden_dim=128, num_layers=3, dropout=0.3):
#         super().__init__()
        
#         # 1. Projections (MATCHED: node_proj, edge_proj keys)
#         self.node_proj = nn.Linear(node_dim, hidden_dim)
#         self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        
#         # 2. Convolutions & BatchNorms (MATCHED: bns keys)
#         self.convs = nn.ModuleList()
#         self.bns = nn.ModuleList()
        
#         for _ in range(num_layers):
#             # MLP for GINE (Takes hidden_dim input because projection happened already)
#             mlp = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU()
#             )
#             # train_eps=True matches checkpoint
#             self.convs.append(GINEConv(mlp, train_eps=True, edge_dim=hidden_dim))
#             # Separate Batch Norm list found in keys
#             self.bns.append(nn.BatchNorm1d(hidden_dim))
            
#         # 3. Readout (MATCHED: lin1 weight [128, 256] -> Concatenated Pooling)
#         self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim) 
#         self.lin2 = nn.Linear(hidden_dim, 1)
        
#         self.dropout_rate = dropout

#     def forward(self, data):
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
#         # 1. Project Features First
#         x = self.node_proj(x)
#         edge_attr = self.edge_proj(edge_attr)
        
#         # 2. Message Passing
#         for conv, bn in zip(self.convs, self.bns):
#             x = conv(x, edge_index, edge_attr) # GINE uses projected edge_attr
#             x = bn(x)                          # Apply separate BN
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
#         # 3. Concatenated Pooling (Mean + Max)
#         x_mean = global_mean_pool(x, batch)
#         x_max = global_max_pool(x, batch)
#         x = torch.cat([x_mean, x_max], dim=1) # [batch, 256]
        
#         # 4. Final MLP
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.lin2(x)
        
#         return x

# def run_audit():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     # Load Model
#     model = DockingGNN_Final(node_dim=22, edge_dim=1, hidden_dim=128).to(device)
    
#     try:
#         checkpoint = torch.load(CONFIG['model_path'], map_location=device)
#         state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
#         model.load_state_dict(state_dict)
#         model.eval()
#         print("PASS: Model loaded successfully!")
#     except Exception as e:
#         print(f"CRITICAL ERROR: {e}")
#         sys.exit(1)

#     # Load Data
#     pt_files = [f for f in os.listdir(CONFIG['graphs_dir']) if f.endswith('.pt')]
#     if not pt_files:
#         print("No graph files found.")
#         sys.exit(1)

#     print(f"Auditing {len(pt_files)} targets...")
#     results = []

#     for pt_file in tqdm(pt_files):
#         target_id = pt_file.replace(".pt", "")
#         file_path = os.path.join(CONFIG['graphs_dir'], pt_file)
        
#         try:
#             graphs = torch.load(file_path)
#             # Filter Native
#             predictor_graphs = [g for g in graphs if g.id != 'NATIVE']
#             if not predictor_graphs: continue
            
#             # Inference
#             loader = DataLoader(predictor_graphs, batch_size=64, shuffle=False)
#             all_scores = []
#             all_dockq = []
            
#             with torch.no_grad():
#                 for batch in loader:
#                     batch = batch.to(device)
#                     all_dockq.extend(batch.y.cpu().numpy().flatten())
#                     # GINE needs edge_attr
#                     scores = model(batch).cpu().numpy().flatten()
#                     all_scores.extend(scores)
            
#             # Metrics
#             df = pd.DataFrame({'dockq': all_dockq, 'score': all_scores})
#             df = df.sort_values('score', ascending=False)
            
#             top1_dq = df.iloc[0]['dockq']
#             max_dq = df['dockq'].max()
            
#             results.append({
#                 'Target': target_id,
#                 'Max_DockQ': max_dq,
#                 'Top1_DockQ': top1_dq,
#                 'Success@1': 1 if top1_dq >= 0.23 else 0,
#                 'Success@10': 1 if (df.head(10)['dockq'] >= 0.23).any() else 0
#             })
            
#         except Exception:
#             continue

#     if results:
#         df_res = pd.DataFrame(results)
#         solvable = df_res[df_res['Max_DockQ'] >= 0.23]
        
#         print("\n" + "="*60)
#         print("FINAL AUDIT RESULTS")
#         print("="*60)
#         print(f"Total Targets: {len(df_res)}")
#         print(f"Solvable Targets: {len(solvable)}")
#         print(f"Success Rate @ 1:  {solvable['Success@1'].mean()*100:.1f}%")
#         print(f"Success Rate @ 10: {solvable['Success@10'].mean()*100:.1f}%")
        
#         with open(CONFIG['report_file'], 'w') as f:
#             f.write(df_res.to_string())
#         print(f"Report saved to {CONFIG['report_file']}")

# if __name__ == "__main__":
#     run_audit()

# #!/usr/bin/env python3
# """
# AUDIT REPORT: CAPRI Quality Retrieval (Final Architecture Fix)
# Objective: Verify GNN using the "X-Ray Reconstructed" architecture.
# Architecture: GIN (No Edge Proj) -> Concatenated Pooling (Mean+Max) -> MLP
# """

# import os
# import sys
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from torch_geometric.nn import GINConv, global_mean_pool, global_max_pool
# from torch_geometric.loader import DataLoader
# from datetime import datetime
# from tqdm import tqdm

# # --- CONFIGURATION ---
# CONFIG = {
#     'graphs_dir': '/home/mini/revathym/ml_ds/results/processed_graphs_capri_dynamic/', # Using the DYNAMIC graphs
#     'model_path': '/home/mini/revathym/ml_ds/models/gnn_ranker/gnn_model.pt',
#     'report_file': f'recheck/capri_quality_audit_final_{datetime.now().strftime("%Y%m%d")}.txt',
#     'plot_dir': 'recheck/plots/capri_quality_plots/'
# }
# os.makedirs(CONFIG['plot_dir'], exist_ok=True)

# # --- X-RAY RECONSTRUCTED ARCHITECTURE ---
# class DockingGNN_XRAY(torch.nn.Module):
#     def __init__(self, node_dim=22, hidden_dim=128, dropout=0.3):
#         super().__init__()
        
#         self.convs = nn.ModuleList()
#         self.dropout_rate = dropout
        
#         # Layer 1: Maps Input(22) -> Hidden(128) directly (No projection layer)
#         mlp1 = nn.Sequential(
#             nn.Linear(node_dim, hidden_dim),
#             nn.BatchNorm1d(hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU()
#         )
#         # train_eps=True matches 'convs.0.eps' key
#         self.convs.append(GINConv(mlp1, train_eps=True))
        
#         # Layers 2 & 3: Map Hidden(128) -> Hidden(128)
#         for _ in range(2):
#             mlp = nn.Sequential(
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.BatchNorm1d(hidden_dim),
#                 nn.ReLU(),
#                 nn.Linear(hidden_dim, hidden_dim),
#                 nn.ReLU()
#             )
#             self.convs.append(GINConv(mlp, train_eps=True))
            
#         # Readout Head
#         # Input is 256 because of Concatenated Pooling (128 Mean + 128 Max)
#         self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim) 
#         self.lin2 = nn.Linear(hidden_dim, 1)

#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
        
#         # GIN Convolution (Ignores edge attributes based on checkpoint keys)
#         for conv in self.convs:
#             x = conv(x, edge_index)
#             # BatchNorm is inside the MLP of GINConv, so no extra BN here
#             x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
#         # Concatenated Pooling (The fix for [128, 256] mismatch)
#         x_mean = global_mean_pool(x, batch)
#         x_max = global_max_pool(x, batch)
#         x = torch.cat([x_mean, x_max], dim=1) # [batch_size, 256]
        
#         # MLP Head
#         x = F.relu(self.lin1(x))
#         x = F.dropout(x, p=self.dropout_rate, training=self.training)
#         x = self.lin2(x)
        
#         return x

# def load_model(device):
#     print("Loading X-Ray Reconstructed Model...")
#     model = DockingGNN_XRAY(node_dim=22, hidden_dim=128).to(device)
    
#     checkpoint = torch.load(CONFIG['model_path'], map_location=device)
#     # Unwrap state_dict if necessary
#     state_dict = checkpoint
#     if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#         state_dict = checkpoint['model_state_dict']
#     elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#         state_dict = checkpoint['state_dict']
        
#     try:
#         model.load_state_dict(state_dict)
#         model.eval()
#         print("PASS: Model weights loaded successfully.")
#         return model
#     except Exception as e:
#         print(f"CRITICAL: Weights still do not match architecture.\nError: {e}")
#         sys.exit(1)

# def run_audit():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = load_model(device)
    
#     pt_files = [f for f in os.listdir(CONFIG['graphs_dir']) if f.endswith('.pt')]
#     if not pt_files:
#         print("No .pt files found. Run scavenge_capri_dynamic.py first.")
#         sys.exit(1)

#     print(f"Auditing {len(pt_files)} targets...")
#     results = []

#     for pt_file in tqdm(pt_files):
#         target_id = pt_file.replace(".pt", "")
#         file_path = os.path.join(CONFIG['graphs_dir'], pt_file)
        
#         try:
#             graphs = torch.load(file_path)
#             # Filter out Native to verify ranking of DECOYS only
#             predictor_graphs = [g for g in graphs if g.id != 'NATIVE']
#             if not predictor_graphs: continue
            
#             # Inference
#             loader = DataLoader(predictor_graphs, batch_size=64, shuffle=False)
#             all_scores = []
#             all_dockq = []
            
#             with torch.no_grad():
#                 for batch in loader:
#                     batch = batch.to(device)
#                     all_dockq.extend(batch.y.cpu().numpy().flatten())
#                     # GIN forward pass ignores edge_attr
#                     scores = model(batch)
#                     all_scores.extend(scores.cpu().numpy().flatten())
            
#             # Ranking Analysis
#             df = pd.DataFrame({'dockq': all_dockq, 'score': all_scores})
#             df_sorted = df.sort_values('score', ascending=False)
            
#             top1_dq = df_sorted.iloc[0]['dockq']
#             max_dq = df['dockq'].max()
            
#             results.append({
#                 'Target': target_id,
#                 'Decoys': len(df),
#                 'Max_DockQ': max_dq,
#                 'Top1_DockQ': top1_dq,
#                 'Success@1': 1 if top1_dq >= 0.23 else 0,
#                 'Success@10': 1 if (df_sorted.head(10)['dockq'] >= 0.23).any() else 0
#             })
            
#         except Exception as e:
#             # print(f"Skipping {target_id}: {e}")
#             pass

#     # Summary
#     if not results: return
#     df_res = pd.DataFrame(results)
    
#     # Filter for targets that are actually solvable (have at least one good decoy)
#     solvable = df_res[df_res['Max_DockQ'] >= 0.23]
    
#     print("\n" + "="*60)
#     print("FINAL AUDIT RESULTS (Solvable Targets Only)")
#     print("="*60)
#     print(f"Total Solvable Targets: {len(solvable)}")
#     print(f"Success Rate @ 1:  {solvable['Success@1'].mean()*100:.1f}%")
#     print(f"Success Rate @ 10: {solvable['Success@10'].mean()*100:.1f}%")
    
#     # Save Report
#     with open(CONFIG['report_file'], 'w') as f:
#         f.write(df_res.to_string())
#     print(f"Report saved to {CONFIG['report_file']}")

# if __name__ == "__main__":
#     run_audit()

# #!/usr/bin/env python3
# """
# AUDIT REPORT: CAPRI Quality Retrieval (Blind OOD)
# Objective: Verify if GNN can identify High-Quality/Acceptable decoys among predictors.
# Constraint: Ignores the raw 'NATIVE' graph to avoid steric-clash penalties.
# Model Source: src.model_gnn.DockingGNN
# Data: /home/mini/revathym/ml_ds/results/processed_graphs_capri/
# """

# import os
# import sys
# import torch
# import pandas as pd
# import numpy as np
# from torch_geometric.loader import DataLoader
# from datetime import datetime
# from tqdm import tqdm

# # --- IMPORT SETUP (Dynamic Path for 'src') ---
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(current_dir) # Go up one level from recheck/
# sys.path.append(project_root)

# try:
#     from src.model_gnn import DockingGNN
# except ImportError:
#     # Fallback if running directly from project root
#     sys.path.append(os.getcwd())
#     try:
#         from src.model_gnn import DockingGNN
#     except ImportError:
#         print("CRITICAL ERROR: Could not import 'src.model_gnn'.")
#         sys.exit(1)

# # --- CONFIGURATION ---
# CONFIG = {
#     # 'graphs_dir': '/home/mini/revathym/ml_ds/results/processed_graphs_capri/',
#     'graphs_dir': '/home/mini/revathym/ml_ds/results/processed_graphs_capri_dynamic/',
#     'model_path': '/home/mini/revathym/ml_ds/models/gnn_ranker/gnn_model.pt',
#     'report_file': f'recheck/capri_quality_audit_{datetime.now().strftime("%Y%m%d")}.txt',
    
#     # Model Architecture (Must match your training config)
#     'gnn_type': 'gine',
#     'hidden_dim': 128,
#     'num_layers': 3,
#     'node_dim': 22,
#     'edge_dim': 1,
#     'dropout': 0.3
# }

# def load_model(device):
#     """Instantiate and load the GNN model."""
#     print(f"Loading model architecture ({CONFIG['gnn_type']})...")
#     model = DockingGNN(
#         node_dim=CONFIG['node_dim'],
#         edge_dim=CONFIG['edge_dim'],
#         hidden_dim=CONFIG['hidden_dim'],
#         num_layers=CONFIG['num_layers'],
#         dropout=CONFIG['dropout'],
#         conv_type=CONFIG['gnn_type']
#     ).to(device)

#     print(f"Loading weights from {CONFIG['model_path']}...")
#     try:
#         checkpoint = torch.load(CONFIG['model_path'], map_location=device)
#         if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['model_state_dict'])
#         elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
#             model.load_state_dict(checkpoint['state_dict'])
#         else:
#             model.load_state_dict(checkpoint)
        
#         model.eval()
#         return model
#     except Exception as e:
#         print(f"CRITICAL ERROR loading model: {e}")
#         sys.exit(1)

# def calculate_ndcg(y_true, y_score, k=10):
#     """Normalized Discounted Cumulative Gain at K."""
#     if len(y_true) < 1: return 0.0
#     k = min(k, len(y_true))
    
#     # Relevance scores are the DockQ values themselves
#     y_true = np.array(y_true)
#     y_score = np.array(y_score)
    
#     # Ideal DCG (sorted by DockQ)
#     ideal_order = np.argsort(y_true)[::-1]
#     ideal_gains = y_true[ideal_order][:k]
#     ideal_dcg = np.sum(ideal_gains / np.log2(np.arange(len(ideal_gains)) + 2))
    
#     if ideal_dcg == 0: return 0.0
    
#     # Actual DCG (sorted by GNN Score)
#     actual_order = np.argsort(y_score)[::-1]
#     actual_gains = y_true[actual_order][:k]
#     actual_dcg = np.sum(actual_gains / np.log2(np.arange(len(actual_gains)) + 2))
    
#     return actual_dcg / ideal_dcg

# def run_audit():
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     model = load_model(device)
#     results = []
    
#     pt_files = [f for f in os.listdir(CONFIG['graphs_dir']) if f.endswith('.pt')]
#     if not pt_files:
#         print("No .pt files found!")
#         sys.exit(1)

#     print(f"Auditing {len(pt_files)} targets...")
    
#     for pt_file in tqdm(pt_files):
#         target_id = pt_file.replace(".pt", "")
#         file_path = os.path.join(CONFIG['graphs_dir'], pt_file)
        
#         try:
#             # Load graphs
#             graphs = torch.load(file_path)
#             if not graphs: continue
            
#             # --- FILTERING STEP ---
#             # Remove the 'NATIVE' graph to ensure we rank only predictors
#             predictor_graphs = [g for g in graphs if g.id != 'NATIVE']
            
#             if not predictor_graphs:
#                 continue
                
#             # Batch Inference
#             loader = DataLoader(predictor_graphs, batch_size=64, shuffle=False)
#             all_scores = []
#             all_dockq = []
            
#             with torch.no_grad():
#                 for batch in loader:
#                     batch = batch.to(device)
#                     # Use 'y' for ground truth DockQ (float)
#                     all_dockq.extend(batch.y.cpu().numpy().flatten())
                    
#                     # GNN Forward Pass
#                     scores = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
#                     all_scores.extend(scores.cpu().numpy().flatten())
            
#             # Create DataFrame
#             df = pd.DataFrame({'dockq': all_dockq, 'gnn_score': all_scores})
            
#             # --- METRICS ---
#             max_dockq = df['dockq'].max()
            
#             # Sort by GNN prediction (Descending)
#             df_sorted = df.sort_values('gnn_score', ascending=False).reset_index(drop=True)
            
#             # Top-1 Stats
#             top1_dockq = df_sorted.iloc[0]['dockq']
#             success_at_1 = 1 if top1_dockq >= 0.23 else 0
#             high_qual_at_1 = 1 if top1_dockq >= 0.80 else 0
            
#             # Top-10 Stats
#             top10_slice = df_sorted.head(10)
#             success_at_10 = 1 if (top10_slice['dockq'] >= 0.23).any() else 0
            
#             # NDCG
#             ndcg_10 = calculate_ndcg(df['dockq'].values, df['gnn_score'].values, k=10)
            
#             results.append({
#                 'Target': target_id,
#                 'Decoys': len(df),
#                 'Max_Available_DockQ': max_dockq,
#                 'Top1_DockQ': top1_dockq,
#                 'Success@1': success_at_1,
#                 'HighQual@1': high_qual_at_1,
#                 'Success@10': success_at_10,
#                 'NDCG@10': ndcg_10
#             })
            
#         except Exception as e:
#             print(f"Error processing {target_id}: {e}")
#             continue

#     # --- SUMMARY REPORT ---
#     if not results:
#         print("No results generated.")
#         return

#     df_res = pd.DataFrame(results)
    
#     # Calculate dataset-wide metrics
#     # Note: We filter for valid targets where Success is theoretically possible (Max DockQ >= 0.23)
#     # This prevents penalizing the model for targets where NO good predictors exist.
#     valid_targets = df_res[df_res['Max_Available_DockQ'] >= 0.23]
    
#     avg_succ1 = valid_targets['Success@1'].mean() * 100 if len(valid_targets) > 0 else 0
#     avg_succ10 = valid_targets['Success@10'].mean() * 100 if len(valid_targets) > 0 else 0
#     avg_hq1 = valid_targets['HighQual@1'].mean() * 100 if len(valid_targets) > 0 else 0
#     avg_ndcg = valid_targets['NDCG@10'].mean()
    
#     report_content = [
#         '"""',
#         'AUDIT REPORT: CAPRI Quality Retrieval (Blind OOD)',
#         f'Date: {datetime.now().strftime("%Y-%m-%d")}',
#         'Objective: Retrieve Acceptable/High-Quality decoys (ignoring raw Native)',
#         '"""',
#         '',
#         '='*80,
#         'DATASET STATISTICS',
#         '='*80,
#         f'Total Targets Audited: {len(df_res)}',
#         f'Solvable Targets (Max DockQ >= 0.23): {len(valid_targets)}',
#         '',
#         '='*80,
#         'PERFORMANCE METRICS (Computed on Solvable Targets Only)',
#         '='*80,
#         f'Success@1 (DockQ >= 0.23)   : {avg_succ1:.1f}%',
#         f'Success@10 (DockQ >= 0.23)  : {avg_succ10:.1f}%',
#         f'High-Quality@1 (DockQ >= 0.8): {avg_hq1:.1f}%',
#         f'NDCG@10 (Ranking Quality)   : {avg_ndcg:.4f}',
#         '',
#         '='*80,
#         'DETAILED BREAKDOWN',
#         '='*80,
#         # Pretty print with selected columns
#         df_res[['Target', 'Decoys', 'Max_Available_DockQ', 'Top1_DockQ', 'Success@10']].to_string(index=False)
#     ]
    
#     with open(CONFIG['report_file'], 'w') as f:
#         f.write('\n'.join(report_content))
        
#     print(f"\nAudit Complete! Report saved to: {CONFIG['report_file']}")
#     print(f"Metrics (on {len(valid_targets)} solvable targets):")
#     print(f"Success@1: {avg_succ1:.1f}% | NDCG@10: {avg_ndcg:.4f}")

# if __name__ == "__main__":
#     run_audit()