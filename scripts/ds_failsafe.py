import freesasa, os, argparse, csv, zipfile, tempfile, sys, logging
import numpy as np
import pandas as pd
from Bio import PDB
from scipy.spatial import KDTree
from concurrent.futures import ProcessPoolExecutor, as_completed


logger = logging.getLogger(__name__)

class SkipComplex(Exception):
    """Signal that this PDB should be skipped and the run should continue."""
    pass


HYDROPHOBIC_RESIDUES = {"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR"}
POSITIVE_RESIDUES = {"LYS", "ARG", "HIS"}
VDW_RADII = {"H": 1.2, "C": 1.7, "N": 1.55, "O": 1.52, "S": 1.8, "P": 1.8}

def identify_interface_residues(complex_pdb, cutoff=7.0):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("complex", complex_pdb)
    chains = list(structure.get_chains())

    if len(chains) < 2:
        print(f"ERROR: Complex '{os.path.basename(complex_pdb)}' has less than 2 chains ({len(chains)}). Skipping.")
        sys.exit(1) # Exit with a non-zero status code to signal a skip/failure

    chain1, chain2 = chains[0], chains[1]

    cb_coords = {
        (chain.id, res.id[1]): (
            res["CB"].coord if "CB" in res else res["CA"].coord if "CA" in res else None
        )
        for chain in [chain1, chain2]
        for res in chain
        if res.id[0] == " " and ("CB" in res or "CA" in res)
    }

    interface_pairs = set()
    for res1 in chain1:
        for res2 in chain2:
            if res1.id[0] == " " and res2.id[0] == " ":
                key1 = (res1.parent.id, res1.id[1])
                key2 = (res2.parent.id, res2.id[1])
                if key1 in cb_coords and key2 in cb_coords:
                    if cb_coords[key1] is not None and cb_coords[key2] is not None:
                        if np.linalg.norm(cb_coords[key1] - cb_coords[key2]) < cutoff:
                            interface_pairs.add(((res1.parent.id, res1), (res2.parent.id, res2)))
    return interface_pairs, chain1.id, chain2.id

def extract_unique_residues(interface_residues):
    unique_residues = set()
    for (chain1, res1), (chain2, res2) in interface_residues:
        unique_residues.add((chain1, res1.id[1], res1.resname))
        unique_residues.add((chain2, res2.id[1], res2.resname))
    return sorted(unique_residues, key=lambda x: (x[0], x[1]))

def compute_interface_surface_area(complex_pdb, unbound_pdbs):
    freesasa.setVerbosity(freesasa.nowarnings)
    structures = [freesasa.Structure(pdb) for pdb in [complex_pdb] + unbound_pdbs]
    sasa_values = [freesasa.calc(struct).totalArea() for struct in structures]
    return sasa_values[1] + sasa_values[2] - sasa_values[0]

def compute_short_contacts(interface_residues):
    total_overlap = 0
    for (chain1, res1), (chain2, res2) in interface_residues:
        for atom1 in res1:
            for atom2 in res2:
                if atom1.element in VDW_RADII and atom2.element in VDW_RADII:
                    r = atom1 - atom2
                    R = VDW_RADII[atom1.element] + VDW_RADII[atom2.element]
                    D = r - (R - 0.40)
                    if D < 0:
                        total_overlap += abs(D)
    return total_overlap

def compute_hydrophobicity_per_monomer(interface_residues):
    monomer1_residues = {res for (chain, res), _ in interface_residues}
    monomer2_residues = {res for _, (chain, res) in interface_residues}
    h1 = sum(res.resname in HYDROPHOBIC_RESIDUES for res in monomer1_residues) / len(monomer1_residues) if monomer1_residues else 0
    h2 = sum(res.resname in HYDROPHOBIC_RESIDUES for res in monomer2_residues) / len(monomer2_residues) if monomer2_residues else 0
    return h1, h2

def compute_spatial_clustering_per_monomer(interface_residues, cutoff=14.0):
    def clustering(residues):
        residue_list = list(residues)
        if len(residue_list) < 2:
            return 0
        cb_coords = [res["CB"].coord if "CB" in res else res["CA"].coord for res in residue_list]
        tree = KDTree(np.array(cb_coords))
        d = len(tree.query_pairs(cutoff))
        return (2 * d) / (len(residue_list) * (len(residue_list) - 1))
    m1 = {res for (chain, res), _ in interface_residues}
    m2 = {res for _, (chain, res) in interface_residues}
    return clustering(m1), clustering(m2)

def compute_scores(decoy_data, use_positive_residue_score=False, use_equal_weights=False, included_parameters=None):
    df = pd.DataFrame(decoy_data)

    # Normalize scores that require normalization
    # Apply normalization conditionally to avoid issues if all values are the same (min == max)
    df['Interface_SA_score'] = df.groupby('Complex')['Interface_SA'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.0
    )
    df['Short_Contacts_Score'] = df['Short_Contacts'].apply(lambda x: np.log1p(x))
    df['Short_Contacts_Score'] = df.groupby('Complex')['Short_Contacts_Score'].transform(
        lambda x: (x.max() - x) / (x.max() - x.min()) if x.max() != x.min() else 0.0
    )
    df['Hydrophobicity_Score'] = (df['Hydrophobicity_Monomer1'] + df['Hydrophobicity_Monomer2']) / 2
    df['Clustering_Score'] = (df['Spatial_Clustering_Monomer1'] + df['Spatial_Clustering_Monomer2']) / 2

    # Define base weights (modified to reflect the original)
    base_weights = {
        'Interface_SA_score': 0.20,
        'Short_Contacts_Score': 0.73,
        'Conserved_Interface_Fraction': 0.05,
        'Hydrophobicity_Score': 0.02,
        'Clustering_Score': 0.00,
        'Positive_Residue_Score': 0.00
    }

    if use_equal_weights:
        # All available parameters get equal weight
        num_active_params = sum(1 for p in base_weights if base_weights[p] > 0)
        equal_weight = 1.0 / num_active_params if num_active_params > 0 else 0
        score_components = {k: equal_weight for k in base_weights}
    else:
        score_components = base_weights.copy()
        score_components['Positive_Residue_Score'] = 1.0 if use_positive_residue_score else 0.0

    # Handle explicitly included parameters (if any)
    if included_parameters:
        for key in list(score_components.keys()):
            if key not in included_parameters:
                score_components[key] = 0.0
    
    # Dynamically adjust weights if parameters cannot be computed
    # This is where robustness for missing conservation comes in
    active_weights = score_components.copy()
    
    # If Conserved_Interface_Fraction was not computable (marked by -1.0)
    # This logic assumes -1.0 means 'not computable'
    if 'Conserved_Interface_Fraction' in df.columns and (df['Conserved_Interface_Fraction'] == -1.0).all():
        print("DEBUG: Conserved_Interface_Fraction not computable for any decoy in this complex. Zeroing its weight.")
        active_weights['Conserved_Interface_Fraction'] = 0.0

    # Ensure Positive_Residue_Score is handled if it's always 0 (no positive residues in interface)
    if 'Positive_Residue_Score' in df.columns and (df['Positive_Residue_Score'] == 0.0).all() and use_positive_residue_score:
        print("DEBUG: Positive_Residue_Score is 0 for all decoys in this complex. Zeroing its weight.")
        active_weights['Positive_Residue_Score'] = 0.0
    
    # Normalize active weights only among those that are actually active
    total_active_weight = sum(active_weights.values())
    if total_active_weight > 0:
        normalized_weights = {k: v / total_active_weight for k, v in active_weights.items()}
    else:
        normalized_weights = {k: 0.0 for k in active_weights}  # All weights become 0 if total is 0

    df['Weighted_Score'] = sum(df[key] * normalized_weights.get(key, 0) for key in df.columns if key in normalized_weights)
    return df.to_dict(orient='records')

def compute_z_scores_per_complex(results, score_key="Weighted_Score"):
    df = pd.DataFrame(results)
    z_scores = df.groupby('Complex')[score_key].transform(lambda x: (x - x.mean()) / x.std(ddof=0))
    df['Z_Score'] = z_scores
    df.sort_values(by=['Complex', 'Z_Score'], ascending=[True, False], inplace=True)
    return df.to_dict(orient='records')

def load_conserved_residues(csv_path):
    df = pd.read_csv(csv_path)
    return set(df["Residue Number"].values)

def analyze_docked_complex(complex_pdb, unbound_pdbs, conserved_rec, conserved_lig):
    try:
        interface_residues, chain_rec, chain_lig = identify_interface_residues(complex_pdb)
    except SkipComplex:
        quit("Skipping complex due to insufficient chains or other issues.")

    unique_interface_residues = extract_unique_residues(interface_residues)
    interface_area = compute_interface_surface_area(complex_pdb, unbound_pdbs)
    short_contacts = compute_short_contacts(interface_residues)
    h1, h2 = compute_hydrophobicity_per_monomer(interface_residues)
    c1, c2 = compute_spatial_clustering_per_monomer(interface_residues)
    total_interface_residues = len(unique_interface_residues)
    positively_charged_count = sum(1 for (_, _, resname) in unique_interface_residues if resname in POSITIVE_RESIDUES)
    positive_residue_score = 1 - (positively_charged_count / total_interface_residues) if total_interface_residues else 0
    conserved_interface = [
        (chain, res_no, resname)
        for (chain, res_no, resname) in unique_interface_residues
        if (chain == chain_rec and res_no in conserved_rec) or (chain == chain_lig and res_no in conserved_lig)
    ]
    conserved_fraction = len(conserved_interface) / total_interface_residues if total_interface_residues else 0
    return {
        # "Complex": os.path.basename(complex_pdb)[:4],
        "Complex": unbound_pdbs[0][:4],
        "Decoy": os.path.basename(complex_pdb),
        "Interface_Residues": unique_interface_residues,
        "Interface_SA": interface_area,
        "Short_Contacts": short_contacts,
        "Hydrophobicity_Monomer1": h1,
        "Hydrophobicity_Monomer2": h2,
        "Spatial_Clustering_Monomer1": c1,
        "Spatial_Clustering_Monomer2": c2,
        "Conserved_Interface_Fraction": conserved_fraction,
        "Positive_Residue_Score": positive_residue_score
    }
# Inside main() function of ds_failsafe.py, when calling analyze_docked_complex
# The chain check needs to happen for the *unbound* receptor and ligand PDBs
# that are passed into analyze_docked_complex or even before that.
# The current check within identify_interface_residues will only apply to the *docked* complex.
# For the case you described ("a complex that has, say, less than 2 chains"),
# this typically refers to the input structure (receptor/ligand) themselves.

# Let's adjust the check to happen earlier, right after loading receptor and ligand paths in main().
# We'll need a helper function.

def check_num_chains(pdb_path):
    parser = PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("temp", pdb_path)
        chains = list(structure.get_chains())
        return len(chains)
    except Exception as e:
        print(f"ERROR: Could not parse PDB file {pdb_path}: {e}")
        return 0 # Indicate an error or 0 chains if parsing fails
    
def main():
    parser = argparse.ArgumentParser(description="Calculate Docking Score (DS) for Web Server Jobs")
    parser.add_argument("--receptor", required=True, help="Path to receptor PDB")
    parser.add_argument("--ligand", required=True, help="Path to ligand PDB")
    parser.add_argument("--receptor-csv", required=True, help="Path to receptor conservation CSV")
    parser.add_argument("--ligand-csv", required=True, help="Path to ligand conservation CSV")
    parser.add_argument("--decoy-zip", required=True, help="Path to zipped decoy pool")
    parser.add_argument("--output", required=True, help="Path to output TSV file")
    parser.add_argument("--max_workers", type=int, default=os.cpu_count(), help="Number of workers")

    args = parser.parse_args()

    # We use basicConfig so it pipes directly into the job_wrapper's job.log
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("Starting DS Failsafe processing...")

    # 1. Prepare Unbound PDBs list required by your function
    unbound_pdbs = [args.receptor, args.ligand]

    # 2. Load Conservation Data using your actual function
    logger.info("Loading conservation CSVs...")
    conserved_rec = load_conserved_residues(args.receptor_csv)
    conserved_lig = load_conserved_residues(args.ligand_csv)

    results = []

    # 3. Safely Extract Decoys
    with tempfile.TemporaryDirectory() as temp_extract_dir:
        logger.info(f"Extracting decoys to temporary secure folder...")
        with zipfile.ZipFile(args.decoy_zip, 'r') as zip_ref:
            zip_ref.extractall(temp_extract_dir)

        decoy_files = []
        for root, _, files in os.walk(temp_extract_dir):
            for file in files:
                if file.endswith('.pdb'):
                    decoy_files.append(os.path.join(root, file))

        logger.info(f"Found {len(decoy_files)} decoys. Starting multiprocessing...")

        # 4. Run the Science using your actual analyze_docked_complex function
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {
                executor.submit(
                    analyze_docked_complex, 
                    decoy_file, 
                    unbound_pdbs, 
                    conserved_rec, 
                    conserved_lig
                ): decoy_file for decoy_file in decoy_files
            }

            for future in as_completed(futures):
                try:
                    res = future.result()
                    if res:
                        results.append(res)
                except Exception as e:
                    # Log silently to avoid spamming the log if a single PDB is malformed
                    pass

    # 5. Save Final Output
    if results:
        df_results = pd.DataFrame(results)
        # Convert Lists/Sets to strings so it saves cleanly to TSV
        if 'Interface_Residues' in df_results.columns:
            df_results['Interface_Residues'] = df_results['Interface_Residues'].astype(str)
            
        df_results.to_csv(args.output, sep='\t', index=False)
        logger.info(f"Successfully saved DS physical features to {args.output}")
    else:
        logger.warning("No successful results were generated!")

if __name__ == "__main__":
    main()