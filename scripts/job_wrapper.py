import os, sys
import argparse
import subprocess
import traceback
from datetime import datetime

def run_step(cmd, log_file, step_name):
    """Helper function to run a subprocess and log it."""
    with open(log_file, "a") as log:
        log.write(f"\n[{datetime.now()}] --- STARTING: {step_name} ---\n")
        try:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            log.write(f"[{datetime.now()}] --- COMPLETED: {step_name} ---\n")
        except subprocess.CalledProcessError as e:
            log.write(f"[{datetime.now()}] --- FATAL ERROR IN {step_name} ---\n")
            raise e

def main(args):

    # --- 1. DYNAMIC JOB PATHS ---
    inputs_dir = os.path.join(args.job_dir, "inputs")
    temp_dir = os.path.join(args.job_dir, "temp")
    outputs_dir = os.path.join(args.job_dir, "outputs")
    log_file = os.path.join(args.job_dir, "job.log")
    
    receptor_path = os.path.join(inputs_dir, args.receptor_name)
    ligand_path = os.path.join(inputs_dir, args.ligand_name)
    
    receptor_base = os.path.splitext(args.receptor_name)[0]
    ligand_base = os.path.splitext(args.ligand_name)[0]

    # Intermediate files generated during the run
    receptor_csv = os.path.join(temp_dir, "conservation", f"{receptor_base}_conserved.csv")
    ligand_csv = os.path.join(temp_dir, "conservation", f"{ligand_base}_conserved.csv")
    ds_output_tsv = os.path.join(temp_dir, "ds_scores.tsv")
    graphs_dir = os.path.join(temp_dir, "graphs")
    final_output_csv = os.path.join(outputs_dir, "predictions.csv")

    # --- 2. STATIC BACKEND PATHS ---
    scripts_dir = os.path.join(args.backend_dir, "scripts")
    pssm_script = os.path.join(scripts_dir, "pssm_conservation.py")
    ds_script = os.path.join(scripts_dir, "ds_failsafe.py")
    graph_script = os.path.join(scripts_dir, "create_graphs_with_edge_features.py")
    inference_script = os.path.join(scripts_dir, "run_blind_inference.py")
    
    blast_bin = os.path.join(args.backend_dir, "blast_backend", "bin", "psiblast")
    blast_db = os.path.join(args.backend_dir, "blast_backend", "db", "swissprot")
    
    xgb_model = os.path.join(args.backend_dir, "models", "xgboost", "XGBoost_regularized_new.joblib")
    gnn_model = os.path.join(args.backend_dir, "models", "gnn", "gnn_with_edge_model_v10.pt")
    ds2_python = os.path.join(args.backend_dir, "ds2_env", "bin", "python")

    # Ensure output directories exist
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    with open(log_file, "a") as log:
        log.write(f"[{datetime.now()}] Initializing Job Wrapper Pipeline...\n")

    # --- EXECUTE STAGE 1: PSSM Conservation ---
    cmd_pssm = [
        ds2_python, pssm_script,
        "--receptor", receptor_path,
        "--ligand", ligand_path,
        "--blast-bin", blast_bin,
        "--blast-db", blast_db,
        "--outdir", temp_dir
    ]
    run_step(cmd_pssm, log_file, "PSSM Generation")

    # Only continue if a decoy pool was provided
    if args.decoy_zip_name:
        decoy_zip_path = os.path.join(inputs_dir, args.decoy_zip_name)
        
        # --- EXECUTE STAGE 2: DS Computation ---
        cmd_ds = [
            ds2_python, ds_script,
            "--receptor", receptor_path,
            "--ligand", ligand_path,
            "--receptor-csv", receptor_csv,
            "--ligand-csv", ligand_csv,
            "--decoy-zip", decoy_zip_path,
            "--output", ds_output_tsv
        ]
        run_step(cmd_ds, log_file, "DS Scoring")

        # --- EXECUTE STAGE 3: Graph Generation ---
        cmd_graphs = [
            ds2_python, graph_script,  # Using ds2_python instead of sys.executable
            "--tsv", ds_output_tsv,
            "--outdir", graphs_dir,
            "--decoy-zip", decoy_zip_path  # <-- This was missing!
        ]
        run_step(cmd_graphs, log_file, "Graph Generation")

        # --- EXECUTE STAGE 4: ML Inference ---
        cmd_inference = [
            ds2_python, inference_script,
            "--tsv", ds_output_tsv,
            "--graph-dir", graphs_dir,
            "--xgb-model", xgb_model,
            "--gnn-model", gnn_model,
            "--output", final_output_csv
        ]
        run_step(cmd_inference, log_file, "ML Inference")
        
        with open(log_file, "a") as log:
            log.write(f"\n[{datetime.now()}] PIPELINE COMPLETE. Results available in {final_output_csv}\n")

def set_status(job_dir, status):
    """Best-effort update of the DS2Job status via Django ORM."""
    try:
        job_id = os.path.basename(job_dir)
        # Bootstrap Django
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
        import django
        django.setup()
        from scoring.models import DS2Job
        DS2Job.objects.filter(job_id=job_id).update(status=status)
    except Exception as e:
        log_file = os.path.join(job_dir, 'job.log')
        with open(log_file, 'a') as f:
            f.write(f"\n[{datetime.now()}] ERROR in set_status: {e}\n")


if __name__ == "__main__":
    # Parse args early so we can access job_dir for status updates
    parser = argparse.ArgumentParser(description="Master Wrapper for a Single DS2 Web Job")
    parser.add_argument("--job-dir", required=True)
    parser.add_argument("--backend-dir", required=True)
    parser.add_argument("--receptor-name", required=True)
    parser.add_argument("--ligand-name", required=True)
    parser.add_argument("--decoy-zip-name", required=False)
    pre_args = parser.parse_args()

    try:
        main(pre_args)
        set_status(pre_args.job_dir, 'COMPLETED')
    except Exception:
        traceback.print_exc()
        log_file = os.path.join(pre_args.job_dir, 'job.log')
        with open(log_file, 'a') as f:
            f.write(f"\n[{datetime.now()}] PIPELINE FAILED\n")
            traceback.print_exc(file=f)
        set_status(pre_args.job_dir, 'FAILED')