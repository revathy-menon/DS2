#!/usr/bin/env python3
"""
Requires NO ground truth labels (DockQ/Fnat). Takes in physical features, filters with XGBoost (0.5), and ranks with GNN.
"""

import os
import sys
import importlib.util
import time
import json
import argparse
from datetime import datetime
import logging
import joblib
import pandas as pd
import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

# We find the architecture folder relative to this script's location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DS2_DIR = os.path.dirname(SCRIPT_DIR) # Go up one level to DS2
GNN_MODELS_DIR = os.path.join(DS2_DIR, 'models', 'gnn')

# Append exactly the gnn models folder to sys.path so standard import works perfectly
if GNN_MODELS_DIR not in sys.path:
    sys.path.append(GNN_MODELS_DIR)

try:
    from gnn_architecture import DockingGNN_Robust
except ImportError as e:
    print(f"❌ ERROR: Could not import DockingGNN_Robust from {GNN_MODELS_DIR}\nDetails: {e}")
    sys.exit(1)

def run_pipeline(input_file, graph_dir, out_dir, xgb_path, gnn_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting Blind Inference Cascade on {device}")
    
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------
    # STAGE 1: Data Loading & XGBoost Filter
    # ---------------------------------------------------------
    print(f"\n--- STAGE 1: XGBoost Filtering ---")
    
    sep = '\t' if input_file.endswith('.tsv') else ','
    df = pd.read_csv(input_file, sep=sep)
    n_total = len(df)
    
    features = [
        'Interface_SA', 'Short_Contacts', 'Hydrophobicity_Monomer1', 
        'Hydrophobicity_Monomer2', 'Spatial_Clustering_Monomer1', 
        'Spatial_Clustering_Monomer2', 'Conserved_Interface_Fraction', 
        'Positive_Residue_Score'
    ]

    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"❌ ERROR: Input file is missing required physical features: {missing_features}")
        sys.exit(1)

    df['Conserved_Interface_Fraction'] = df['Conserved_Interface_Fraction'].fillna(0.0)
        
    X = df[features].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    
    xgb_model = joblib.load(xgb_path)
    
    # Provide the exact probability score
    df['XGBoost_Probability'] = xgb_model.predict_proba(X)[:, 1]
    
    # Provide the CAPRI quality classification based on the 0.5 threshold
    df['CAPRI_Quality_Prediction'] = np.where(
        df['XGBoost_Probability'] >= 0.5, 
        'Predicted Acceptable+', 
        'Predicted Incorrect'
    )
    
    survivors = df[df['XGBoost_Probability'] >= 0.5].copy()
    failed = df[df['XGBoost_Probability'] < 0.5].copy()

    # ---------------------------------------------------------
    # STAGE 2: GNN Re-Ranking
    # ---------------------------------------------------------
    print(f"\n--- STAGE 2: GNN Re-Ranking ---")
    
    survivors['GNN_Score'] = -999.0 
    failed['GNN_Score'] = -1000.0 # Force failed decoys to the bottom
    
    if len(survivors) > 0:
        device = torch.device('cpu')
        
        gnn_model = DockingGNN_Robust(node_dim=22, edge_dim=4, hidden_dim=64).to(device)
        gnn_model.load_state_dict(torch.load(gnn_path, map_location=device, weights_only=False), strict=True)
        gnn_model.eval()
        
        target_graphs = []
        for decoy_name in survivors['Decoy'].values:
            pt_file = os.path.join(graph_dir, f"{decoy_name}.pt")
            if os.path.exists(pt_file):
                graph = torch.load(pt_file, weights_only=False)
                graph.id = decoy_name 
                target_graphs.append(graph)
                
        if target_graphs:
            loader = DataLoader(target_graphs, batch_size=64, shuffle=False)
            graph_ids = []
            graph_scores = []
            
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    batch.edge_attr[:, 0] = torch.clamp(batch.edge_attr[:, 0], max=1.0) 
                    batch.edge_attr[:, 1] = 0.0 
                    batch.edge_attr[:, 3] = torch.clamp(batch.edge_attr[:, 3], min=-2.0, max=2.0) 
                    
                    scores = gnn_model(batch).cpu().numpy().flatten()
                    graph_scores.extend(scores)
                    if isinstance(batch.id, list):
                        graph_ids.extend(batch.id)
                    else:
                        graph_ids.extend(batch.id.tolist())
            
            score_dict = dict(zip(graph_ids, graph_scores))
            survivors['GNN_Score'] = survivors['Decoy'].map(score_dict).fillna(-999.0)

    # ---------------------------------------------------------
    # STAGE 3: Final Sorting & Saving
    # ---------------------------------------------------------
    print(f"\n--- STAGE 3: Saving Final Rankings ---")
    
    final_df = pd.concat([survivors, failed], ignore_index=True)
    
    # Sort primarily by GNN Score, secondarily by XGBoost probability
    final_df = final_df.sort_values(by=['GNN_Score', 'XGBoost_Probability'], ascending=[False, False])
    
    # Assign integer ranks
    final_df['Rank'] = range(1, len(final_df) + 1)
    
    final_df['GNN_Score'] = final_df['GNN_Score'].astype(object)
    final_df.loc[final_df['GNN_Score'] == -1000.0, 'GNN_Score'] = 'NA'
    
    # Clean up the output columns for the user
    output_cols = ['Rank', 'Decoy', 'CAPRI_Quality_Prediction', 'XGBoost_Probability', 'GNN_Score']
    output_cols = [col for col in output_cols if col in final_df.columns]
    
    out_csv = os.path.join(out_dir, 'blind_predictions.csv')
    final_df[output_cols].to_csv(out_csv, index=False, float_format='%.4f')
    
    print(f"\n SCoring complete. Predictions saved to: {out_csv}")

def main():
    parser = argparse.ArgumentParser(description="Run Blind Inference (XGBoost + GNN)")
    parser.add_argument("--tsv", required=True, help="Path to input DS scores TSV")
    parser.add_argument("--graph-dir", required=True, help="Directory containing pre-processed .pt graphs")
    parser.add_argument("--xgb-model", required=True, help="Path to pre-trained XGBoost joblib model")
    parser.add_argument("--gnn-model", required=True, help="Path to pre-trained GNN .pt model")
    parser.add_argument("--output", required=True, help="Path to save final predictions.csv")
    
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output)

    run_pipeline(
        input_file=args.tsv,
        graph_dir=args.graph_dir,
        out_dir=out_dir,
        xgb_path=args.xgb_model,
        gnn_path=args.gnn_model
    )
    
    import shutil
    default_out = os.path.join(out_dir, 'blind_predictions.csv')
    if os.path.exists(default_out) and default_out != args.output:
        shutil.move(default_out, args.output)

if __name__ == "__main__":
    main()
