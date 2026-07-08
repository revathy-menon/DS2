import os
import argparse
import subprocess
from Bio import PDB, SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
from Bio.Data.IUPACData import protein_letters_3to1

def extract_sequence_from_pdb(pdb_file, fasta_file):
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    sequence = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] == " ":
                    resname = residue.resname.capitalize()
                    if resname in protein_letters_3to1:
                        sequence.append(protein_letters_3to1[resname])
                    else:
                        print(f"Warning: Skipping unknown residue {resname} at {residue.id}")

    seq_str = "".join(sequence)
    if not seq_str:
        raise ValueError(f"No valid amino acid sequence found in {pdb_file}.")

    seq_record = SeqRecord(Seq(seq_str), id="PDB_Sequence", description=f"Extracted from {pdb_file}")
    SeqIO.write(seq_record, fasta_file, "fasta")
    print(f"Sequence extracted and saved to {fasta_file}")

def run_psiblast(blast_bin, fasta_file, blast_db, pssm_file, num_threads=16, iterations=2, evalue=0.001, max_seqs=350):
    cmd = [
        blast_bin,  # Dynamic path instead of hardcoded
        "-num_threads", str(num_threads),
        "-query", fasta_file,
        "-db", blast_db,
        "-num_iterations", str(iterations),
        "-evalue", str(evalue),
        "-max_target_seqs", str(max_seqs),
        "-out_ascii_pssm", pssm_file,
        "-out", f"{fasta_file}.psiblast.out"
    ]
    subprocess.run(cmd, check=True)
    print(f"PSI-BLAST completed for {fasta_file}")

def extract_conserved_residues(pssm_file, output_csv):
    with open(pssm_file, "r") as f:
        lines = f.readlines()

    scores = []
    read_flag = False
    amino_acids = list("ARNDCQEGHILKMFPSTWYV")

    for line in lines:
        parts = line.strip().split()

        if parts and parts[0].isdigit() and len(parts) > 42:
            read_flag = True

        if read_flag and len(parts) > 42:
            res_num = int(parts[0])
            query_aa = parts[1]
            log_scores = list(map(int, parts[2:22]))

            max_log_value = max(log_scores)
            max_log_aa = amino_acids[log_scores.index(max_log_value)]

            if max_log_aa == query_aa and max_log_value > 5:
                scores.append((res_num, query_aa, max_log_value))

    df = pd.DataFrame(scores, columns=["Residue Number", "Query AA", "log Value"])
    df.to_csv(output_csv, index=False)
    print(f"Conserved residues saved to {output_csv}")

def process_pdb_file(pdb_path, blast_bin, blast_db, output_dir):
    base_name = os.path.splitext(os.path.basename(pdb_path))[0]
    fasta_dir = os.path.join(output_dir, "fasta")
    pssm_dir = os.path.join(output_dir, "pssm")
    csv_dir = os.path.join(output_dir, "conservation")

    os.makedirs(fasta_dir, exist_ok=True)
    os.makedirs(pssm_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    fasta_file = os.path.join(fasta_dir, f"{base_name}.fasta")
    pssm_file = os.path.join(pssm_dir, f"{base_name}.pssm")
    csv_file = os.path.join(csv_dir, f"{base_name}_conserved.csv")

    extract_sequence_from_pdb(pdb_path, fasta_file)
    run_psiblast(blast_bin, fasta_file, blast_db, pssm_file)
    extract_conserved_residues(pssm_file, csv_file)

def main():
    parser = argparse.ArgumentParser(description="Process ligand and receptor PDBs to get conserved residues.")
    parser.add_argument("--receptor", required=True, help="Path to receptor PDB file")
    parser.add_argument("--ligand", required=True, help="Path to ligand PDB file")
    parser.add_argument("--blast-bin", required=True, help="Path to psiblast executable")
    parser.add_argument("--blast-db", required=True, help="Path to BLAST swissprot database")
    parser.add_argument("--outdir", required=True, help="Base output directory (the job's temp folder)")

    args = parser.parse_args()

    for pdb_file in [args.receptor, args.ligand]:
       if os.path.exists(pdb_file):
           print(f"Processing {pdb_file}")
           process_pdb_file(pdb_file, args.blast_bin, args.blast_db, args.outdir)
       else:
           print(f"Warning: File not found - {pdb_file}")

if __name__ == "__main__":
    main()