import os
import io
import ast
import time
import logging
import zipfile
import torch
import argparse
import numpy as np
import pandas as pd
import biotite.structure as struc
import biotite.structure.io.pdb as pdb
from torch_geometric.data import Data
from joblib import Parallel, delayed
from tqdm import tqdm

# trying to limit cores here
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# node features
AA_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4, 'GLN': 5, 'GLU': 6, 'GLY': 7, 
    'HIS': 8, 'ILE': 9, 'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14, 
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}
HYDROPATHY_ARR = np.array([1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5, 3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2]) 
CHARGE_ARR = np.array([0, 1, 0, -1, 0, 0, -1, 0, 0.5, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

def parse_interface_string(s):
    try:
        res_list = ast.literal_eval(s)
        return set((r[0], r[1]) for r in res_list)
    except:
        return set()

def get_physics_edges(atoms, ca_atoms, edge_cutoff=6.0):
    """
    Computes rich physical edge features using ALL atoms, 
    but maps them back to the CA-CA residue graph.
    """
    
    # Build adjacency matrix using CA atoms
    cell_list_ca = struc.CellList(ca_atoms, cell_size=edge_cutoff)
    adj = cell_list_ca.create_adjacency_matrix(edge_cutoff)
    rows, cols = adj.nonzero()
    
    # Remove self-loops
    mask = rows != cols
    rows, cols = rows[mask], cols[mask]
    edge_index = np.stack([rows, cols])
    
    edge_attrs = []
    
    # for fast lookups for all atoms belonging to a specific residue
    # Create an index mapping: CA array index -> (chain_id, res_id)
    res_identifiers = [(at.chain_id, at.res_id) for at in ca_atoms]
    
    # Pre-group all atoms by residue for fast physics calculation
    res_to_atoms = {}
    for c_id, r_id in set(res_identifiers):
        res_to_atoms[(c_id, r_id)] = atoms[(atoms.chain_id == c_id) & (atoms.res_id == r_id)]

    for i, j in zip(rows, cols):
        res_i_id = res_identifiers[i]
        res_j_id = res_identifiers[j]
        
        atoms_i = res_to_atoms.get(res_i_id)
        atoms_j = res_to_atoms.get(res_j_id)
        
        if atoms_i is None or atoms_j is None or len(atoms_i) == 0 or len(atoms_j) == 0:
             # Fallback if something is weird
             edge_attrs.append([0.0, 0.0, 0.0, 0.0])
             continue

        # Calculate all pairwise distances between atoms of Residue I and Residue J
        # Fast vectorized distance calculation
        coords_i = atoms_i.coord[:, np.newaxis, :]
        coords_j = atoms_j.coord[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(coords_i - coords_j, axis=2)
        
        # Feature 1: Minimum Atomic Distance
        min_dist = dist_matrix.min()
        
        # Feature 2: Steric Clashes (Count of atoms too close, < 2.2A)
        # Note: In a real decoy, H-atoms might not be present. 2.2A is a safe VDW overlap threshold for heavy atoms.
        clash_count = np.sum(dist_matrix < 2.2)
        
        # Feature 3: Hydrogen Bond Proxy (N-O or O-N distance < 3.5A)
        n_mask_i = atoms_i.element == "N"
        o_mask_i = atoms_i.element == "O"
        n_mask_j = atoms_j.element == "N"
        o_mask_j = atoms_j.element == "O"
        
        h_bond = 0.0
        # Check N(i) to O(j)
        if np.any(n_mask_i) and np.any(o_mask_j):
            sub_dist = dist_matrix[np.ix_(n_mask_i, o_mask_j)]
            if sub_dist.size > 0 and sub_dist.min() < 3.5:
                h_bond = 1.0
        # Check O(i) to N(j)
        if np.any(o_mask_i) and np.any(n_mask_j):
            sub_dist = dist_matrix[np.ix_(o_mask_i, n_mask_j)]
            if sub_dist.size > 0 and sub_dist.min() < 3.5:
                h_bond = 1.0
                
        # Feature 4: Coulombic Electrostatics Proxy
        # Retrieve charge from your global CHARGE_ARR using the residue name
        idx_i = AA_MAP.get(ca_atoms[i].res_name, -1)
        idx_j = AA_MAP.get(ca_atoms[j].res_name, -1)
        charge_i = CHARGE_ARR[idx_i] if idx_i != -1 else 0.0
        charge_j = CHARGE_ARR[idx_j] if idx_j != -1 else 0.0
        
        # Q1 * Q2 / min_dist
        coulombic = (charge_i * charge_j) / (min_dist + 1e-5)
        
        edge_attrs.append([
            1.0 / (min_dist + 1e-5), # Inverse min distance
            float(clash_count),      # Steric penalty
            h_bond,                  # Hydrogen bond reward
            coulombic                # Electrostatic interaction
        ])

    edge_index = torch.from_numpy(edge_index).long()
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    
    return edge_index, edge_attr

def pdb_to_graph_physics(atoms, pdb_id, target_residues=None, edge_cutoff=6.0):
    """Generates a graph with 4-dimensional edge features"""
    try:
       
        # Filter amino acids first to remove waters/ligands
        all_atoms = atoms[struc.filter_amino_acids(atoms)]
        ca_atoms = all_atoms[all_atoms.atom_name == "CA"]
        
        if len(ca_atoms) == 0: return None
        
        # INTERFACE MASK: 
        # If target_residues are provided, filter the nodes.
        # If not provided, use all CA atoms in the complex.
        if target_residues and len(target_residues) > 0:
             mask = np.array([(a.chain_id, a.res_id) in target_residues for a in ca_atoms], dtype=bool)
             node_atoms = ca_atoms[mask]
        else:
             node_atoms = ca_atoms
        
        if len(node_atoms) < 5: return None
            
        # Node features
        aa_indices = np.array([AA_MAP.get(r, -1) for r in node_atoms.res_name])
        valid = aa_indices != -1
        node_atoms = node_atoms[valid]
        aa_indices = aa_indices[valid]
        
        if len(node_atoms) == 0: return None

        x = torch.tensor(np.hstack([
            np.eye(20)[aa_indices], 
            HYDROPATHY_ARR[aa_indices].reshape(-1, 1), 
            CHARGE_ARR[aa_indices].reshape(-1, 1)
        ]), dtype=torch.float)
        
        # Calculate edge features
        edge_index, edge_attr = get_physics_edges(all_atoms, node_atoms, edge_cutoff)
        
        # Provide a dummy 'y' tensor (0.0) so PyG DataLoaders don't break
        dummy_y = torch.tensor([0.0], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, 
                    y=dummy_y, id=pdb_id)
    except Exception as e:
        # logging.warning(f"Graph Gen Error {pdb_id}: {e}")
        return None

def process_single_job(df, zip_path, out_dir):
    """
    Processes the single uploaded zip file for the current job.
    Saves the graphs as individual .pt files.
    """
    if not os.path.exists(zip_path):
        logging.error(f"Missing ZIP at provided path: {zip_path}")
        return 0

    processed_graphs = []
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            zip_contents = set(z.namelist())
            
            for _, row in df.iterrows():
                decoy_name = row['Decoy']
                
                # Check for the file exactly as named, or inside a subfolder
                target_file = None
                for candidate in [decoy_name, f"{decoy_name}.pdb", f"*/{decoy_name}", f"*/{decoy_name}.pdb"]:
                    matches = [name for name in zip_contents if name.endswith(decoy_name) or name.endswith(f"{decoy_name}.pdb")]
                    if matches:
                        target_file = matches[0]
                        break
                
                if not target_file:
                    logging.warning(f"Could not find decoy {decoy_name} in zip.")
                    continue
                
                with z.open(target_file) as f:
                    text_stream = io.TextIOWrapper(f, encoding='utf-8')
                    pdb_file = pdb.PDBFile.read(text_stream)
                    structure = pdb.get_structure(pdb_file, model=1)
                    
                    # BLIND INFERENCE: Check if Interface_Residues exists in the TSV.
                    # If not, pass None and the graph will use all residues.
                    target_res = None
                    if 'Interface_Residues' in row and pd.notna(row['Interface_Residues']):
                        target_res = parse_interface_string(row['Interface_Residues'])
                    
                    # Generate the graph
                    graph = pdb_to_graph_physics(structure, decoy_name, target_residues=target_res)
                    
                    if graph is not None:
                        graph.decoy_name = decoy_name  # Save name for inference identification
                        processed_graphs.append(graph)
                        
                        # Save individually
                        pt_file = os.path.join(out_dir, f"{decoy_name}.pt")
                        torch.save(graph, pt_file)
        
        return len(processed_graphs)

    except Exception as e:
        logging.error(f"Failed to process job zip: {str(e)}")
        return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PyG graphs from DS output for Web Server")
    parser.add_argument("--tsv", required=True, help="Path to input DS scores TSV")
    parser.add_argument("--decoy-zip", required=True, help="Path to the user's uploaded decoy zip file")
    parser.add_argument("--outdir", required=True, help="Path to save generated .pt files")
    
    args = parser.parse_args()

    # Log to stdout so the job_wrapper catches it
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    os.makedirs(args.outdir, exist_ok=True)
    
    logging.info(f"Loading DS scores from: {args.tsv}")
    try:
        # Load the TSV generated by the previous DS step
        df = pd.read_csv(args.tsv, sep='\t')
    except Exception as e:
        logging.error(f"Failed to read TSV: {e}")
        exit(1)
    
    logging.info(f"Processing single job decoy zip: {args.decoy_zip}")
    
    # Run the processing synchronously for this one job
    num_graphs = process_single_job(df, args.decoy_zip, args.outdir)
    
    logging.info(f"Graph generation complete. Successfully created {num_graphs} graphs in {args.outdir}")

