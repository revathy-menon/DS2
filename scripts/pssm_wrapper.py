import os
import subprocess
from tqdm import tqdm
from datetime import datetime

# Set base paths
project_dir = "/home/gendis/revathy"  # <-- Update this
pdb_folder = os.path.join(project_dir, "pdbs/capri")
blast_db = os.path.join(project_dir, "swissprot", "swissprot_db")  # Or whatever filename it has
script_path = os.path.join(project_dir, "scripts", "pssm_conservation.py")
output_dir = os.path.join(project_dir, "output_capri")
log_dir = os.path.join(output_dir, "logs")
#print(pdb_folder)
#print(output_dir)
#print(log_dir)
#quit()
os.makedirs(log_dir, exist_ok=True)

# Gather PDB IDs
all_files = os.listdir(pdb_folder)
#base_ids = {f.split("_")[0] for f in all_files if f.endswith(("_A.pdb", "_B.pdb"))}
# base_ids = {f.split("_")[0] for f in all_files if f.endswith((".pdb"))}
# base_ids = {f.split("_")[0] for f in all_files if f.endswith(("_r_b.pdb", "_l_b.pdb"))}
base_ids = {f.split("_")[0] for f in all_files if f.endswith(("_r_u.pdb", "_l_u.pdb"))}
base_ids = sorted(base_ids)

# Summary log
summary_log = os.path.join(log_dir, f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

with open(summary_log, "w") as summary:
    summary.write(f"PSI-BLAST Run Summary - {datetime.now()}\n")
    summary.write("="*50 + "\n\n")

    print(f"🔍 Found {len(base_ids)} unique PDB IDs to process.\n")

    for base_id in tqdm(base_ids, desc="Processing PDBs", unit="file"):
        log_file = os.path.join(log_dir, f"{base_id}.log")
        cmd = [
            "python3", script_path,
            "-id", base_id,
            "-db", blast_db,
            "--outdir", output_dir
        ]
        try:
            with open(log_file, "w") as log:
                subprocess.run(cmd, cwd=os.path.join(project_dir, "scripts"), stdout=log, stderr=log, check=True)
            summary.write(f"[SUCCESS] {base_id}\n")
        except subprocess.CalledProcessError:
            summary.write(f"[FAILED ] {base_id} — See {log_file}\n")
