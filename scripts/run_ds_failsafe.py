import os, time
import subprocess
import csv
import argparse
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

#-------------------------------------------
# USAGE: python3 run_ds_failsafe.py --max_workers 6 --skip_existing
#-------------------------------------------

def get_all_pdb_ids(pdbs_folder):
    pdb_ids = []
    for filename in os.listdir(pdbs_folder):
        if filename.endswith("_u.pdb"):
            pdb_id = filename.split("_")[0]
            pdb_ids.append(pdb_id.lower())
    return sorted(set(pdb_ids))


def run_scoring(pdb_id, base_dir, skip_existing):
    data_dir = os.path.join(base_dir, "capri_formatted_for_ds")
    output_dir = os.path.join(data_dir, "output")
    log_dir = os.path.join(data_dir, "logs")
    script_path = os.path.join(base_dir, "scripts", "ds_failsafe.py")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    zip_path = os.path.join(data_dir, "zipped", f"{pdb_id.upper()}.zip")
    receptor_path = os.path.join(data_dir, "pdbs", f"{pdb_id.upper()}_r_u.pdb")
    ligand_path = os.path.join(data_dir, "pdbs", f"{pdb_id.upper()}_l_u.pdb")
    
    receptor_csv = os.path.join(data_dir, "conservation", f"{pdb_id.upper()}_r_u_conserved.csv")
    ligand_csv = os.path.join(data_dir, "conservation", f"{pdb_id.upper()}_l_u_conserved.csv")

    output_path = os.path.join(output_dir, f"{pdb_id.upper()}_scores.tsv")

    if skip_existing and os.path.exists(output_path):
        return (pdb_id, "SKIPPED", "")

    # Warn if either conservation file is missing
    if not os.path.exists(receptor_csv):
        print(f"WARNING: Receptor conservation file not found: {receptor_csv}. Will treat receptor conservation as empty.")
    if not os.path.exists(ligand_csv):
        print(f"WARNING: Ligand conservation file not found: {ligand_csv}. Will treat ligand conservation as empty.")

    command = [
        "python3", script_path,
        "-z", zip_path,
        "-rec", receptor_path,
        "-lig", ligand_path,
        "-csv_rec", receptor_csv,
        "-csv_lig", ligand_csv,
        "-o", output_path,
        "--use_positive_residues",
      
    try:
        print(f"Running scoring for {pdb_id}...\nCommand: {' '.join(command)}")
        with tqdm(total=1, desc=f"{pdb_id}", position=1, leave=False, bar_format='{desc:<10} {percentage:3.0f}%|{bar}|') as pbar:
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            pbar.update(1)
        return (pdb_id, "SUCCESS", "")
    except subprocess.CalledProcessError as e:
        error_msg = f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}"
        print(f"FAILED {pdb_id}:\n{error_msg}")
        return (pdb_id, "FAILED", error_msg)


def main():
    parser = argparse.ArgumentParser(description="Run capsdock scoring pipeline on all PDBs.")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel processes to run")
    parser.add_argument("--skip_existing", action="store_true", help="Skip PDBs with existing output")
    args = parser.parse_args()

    base_dir = "." # <--- UPDATE THIS
    data_dir = os.path.join(base_dir, "pdb_folder") # <--- UPDATE THIS
    pdbs_folder = os.path.join(data_dir, "pdbs") # <--- UPDATE THIS
    log_dir = os.path.join(data_dir, "logs") # <--- UPDATE THIS

    os.makedirs(log_dir, exist_ok=True)

    pdb_ids = get_all_pdb_ids(pdbs_folder)
    if not pdb_ids:
        print("No PDB IDs found in 'pdbs' folder.")
        return

    print(f"Processing {len(pdb_ids)} PDB(s) using {args.max_workers} workers...\n")

    summary = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(run_scoring, pdb_id, base_dir, args.skip_existing): pdb_id
            for pdb_id in pdb_ids
        }

        # Overall progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Overall Progress", position=0):
            pdb_id = futures[future]
            try:
                pdb_id, status, error = future.result()
                summary.append({"PDB ID": pdb_id, "Status": status, "Error": error})
            except Exception as exc:
                summary.append({"PDB ID": pdb_id, "Status": "FAILED", "Error": str(exc)})

    # Write logs
    with open(os.path.join(log_dir, "success.log"), "a") as slog, \
         open(os.path.join(log_dir, "failure.log"), "a") as flog:
        for entry in summary:
            timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
            if entry["Status"] == "SUCCESS":
                slog.write(f"{timestamp} {entry['PDB ID']} - SUCCESS\n")
            elif entry["Status"] == "FAILED":
                flog.write(f"{timestamp} {entry['PDB ID']} - FAILED - {entry['Error']}\n")

    # Write summary CSV
    summary_csv = os.path.join(log_dir, "summary.csv")
    with open(summary_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["PDB ID", "Status", "Error"])
        writer.writeheader()
        writer.writerows(summary)

    print(f"\n📋 Summary written to: {summary_csv}")


if __name__ == "__main__":
    main()
