# DockScore 2.0 (DS2)

**DockScore 2.0** is a protein–protein docking scoring function that combines an XGBoost ensemble with a Graph Neural Network (GNN) to rank structural decoys of protein complexes. It is the main computational backend of the [SODAPOP](https://sodapop.example.com) web server.

Given a receptor PDB, a ligand PDB, and a set of docked decoy structures (as a ZIP archive of PDB files), DS2 computes per-decoy quality scores and returns a ranked predictions CSV.

---

## Contents

```
DS2/
├── README.md
├── environment.yml          ← conda environment (all dependencies)
├── scripts/
│   ├── pssm_conservation.py          ← Stage 1: PSSM + evolutionary conservation
│   ├── ds_failsafe.py                ← Stage 2: docking score feature extraction
│   ├── create_graphs_with_edge_features.py  ← Stage 3: graph construction
│   ├── run_blind_inference.py        ← Stage 4: XGBoost + GNN inference
│   └── job_wrapper.py               ← Top-level pipeline runner
└── models/
    ├── xgboost/
    │   └── XGBoost_regularized_new.joblib
    └── gnn/
        └── gnn_with_edge_model_v10.pt
```

> **BLAST dependency:** Stage 1 requires `psiblast` and a local copy of the SwissProt database. See [Installation](#installation) for setup instructions.

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.12.3 |
| PyTorch | 2.10.0 |
| PyTorch Geometric | 2.7.0 |
| XGBoost | 3.2.0 |
| NumPy | 2.4.3 |
| Pandas | 3.0.1 |
| SciPy | 1.17.1 |
| NetworkX | 3.6.1 |
| Biopython | 1.86 |
| Joblib | 1.5.3 |
| BLAST+ (psiblast) | ≥ 2.13 |

All Python dependencies are listed in `environment.yml` and can be installed with conda (see below).

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/DS2.git
cd DS2
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate ds2
```

This installs all Python dependencies including PyTorch, PyTorch Geometric, XGBoost, and Biopython.

### 3. Install BLAST+

BLAST+ is required for the PSSM conservation stage. The easiest way is via conda:

```bash
conda activate ds2
conda install -c bioconda blast
```

Then download and format the SwissProt database:

```bash
# Create a directory for the database
mkdir -p blast_backend/db
cd blast_backend/db

# Download SwissProt FASTA
wget https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_sprot.fasta.gz

# Build the BLAST database
makeblastdb -in uniprot_sprot.fasta -dbtype prot -out swissprot

cd ../..
```

---

## Usage

### Run the full pipeline

```bash
python scripts/job_wrapper.py \
  --job-dir     /path/to/output_directory \
  --backend-dir /path/to/DS2 \
  --receptor-name  receptor.pdb \
  --ligand-name    ligand.pdb \
  --decoy-zip-name decoys.zip
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--job-dir` | Output directory (will be created if it does not exist). Subdirectories `inputs/`, `temp/`, and `outputs/` are created automatically. |
| `--backend-dir` | Path to the DS2 root directory (where `scripts/` and `models/` live). |
| `--receptor-name` | Filename of the receptor PDB (must be placed in `<job-dir>/inputs/` before running). |
| `--ligand-name` | Filename of the ligand PDB (same directory). |
| `--decoy-zip-name` | Filename of the decoy ZIP archive (same directory). |

### Input format

- **Receptor / Ligand PDB**: Standard PDB format. Must contain `ATOM` records. Single-chain structures are recommended.
- **Decoy ZIP**: A ZIP archive containing one or more `.pdb` files, each representing a docked conformation of the complex.

### Output

Results are written to `<job-dir>/outputs/predictions.csv`. Columns include per-decoy scores from the XGBoost model, the GNN model, and the ensemble prediction.
---
Example files are provided in the mock_media folder.

---

## Pipeline overview

```
Stage 1 — PSSM Conservation  (pssm_conservation.py)
    ↓  PSI-BLAST against SwissProt → per-residue conservation scores
Stage 2 — Feature Extraction  (ds_failsafe.py)
    ↓  Interface geometry, electrostatics, SASA, contact features
Stage 3 — Graph Construction  (create_graphs_with_edge_features.py)
    ↓  Residue-level contact graph with edge features
Stage 4 — Inference           (run_blind_inference.py)
    ↓  XGBoost + GNN → ensemble score → predictions.csv
```

---

## Citation

If you use DockScore 2.0 in your research, please cite:

> *[Manuscript in preparation — citation will be added upon publication]*

---

## License

This project is released for academic and non-commercial use. Please contact the authors for commercial licensing.
