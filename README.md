# DQN MIP for Disease Network Testing

This repository adapts the DQN-MIP method for disease testing on network graphs with frontier constraints.

## Key Adaptations

### 1. **Graph-Based Environment** (`environment/`)

The environment system consists of several key components:

- **`frontier_batch_env.py`** - `BinaryFrontierEnvBatch`: Core environment implementing frontier constraints

- **`abstract_joint_probability_class.py`**: Abstract base class for probability distributions

- **`log_junction_tree.py`** - `LogJunctionTree`: Efficient probabilistic inference

- **`disease_graph_loader.py`** - Data loading and environment creation

- **`ICPSR_22140_processor.py`** - `ICPSR22140Processor`: Processes raw ICPSR dataset

- **`graph_utils.py`** - Utility functions for graph visualization 

**How they work together**: `driver_disease.py` uses `disease_graph_loader.py` to load data and create the environment. The environment uses `LogJunctionTree` to compute probabilities for the frontier-based testing process.

### 2. **Algorithms** (`algos/`)

- **`dqn_estimator_disease.py`** - DQN implementation for graph environments

- **`baselines_disease.py`** - Baseline algorithms

- **`baseline_iterative_dqn_disease.py`** - Iterative DQN baseline

- **`evaluate_MIP_disease.py`** - DQN-MIP evaluation

### 3. **Graph-Aware MILP Approximator** (`approximator/batch_graph_approximator.py`)

### 4. **Main Driver** (`driver_disease.py`)

**Usage**: Run via `./run.sh` or directly: `python driver_disease.py -D HIV -T 300 -B 5 -V 10`

## Setup

### 1. Create Conda Environment
```bash
conda create -n AFEG python=3.12
conda activate AFEG
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download ICPSR Data
1. Download ICPSR dataset 22140 from https://www.icpsr.umich.edu/web/ICPSR/studies/22140
2. Extract to `ICPSR_22140/` directory with structure:
   ```
   ICPSR_22140/
   ├── DS0001/22140-0001-Data.tsv
   ├── DS0002/22140-0002-Data.tsv
   └── DS0003/22140-0003-Data.tsv
   ```

### 4. Configure Gurobi License
- Place `gurobi.lic` in the repository root, OR
- Set `GRB_LICENSE_FILE` environment variable, OR
- For compute nodes, set `GUROBI_LICENSE_SERVER="PORT@SERVER"`

## Usage

### Local Run
```bash
# Quick test (THRESHOLD=30, BUDGET=3, N_EVAL=2)
./run.sh 30 3 0 2

# Full experiment (THRESHOLD=300, BUDGET=5, N_EVAL=10)
./run.sh 300 5 0 10
```

### SLURM Job
```bash
# Adjust paths in run_job.sbatch first
sbatch run_job.sbatch [THRESHOLD] [BUDGET] [SEED] [N_EVAL]
```

### Parameters
- `THRESHOLD`: Minimum nodes in sampled connected components (default: 300)
- `BUDGET`: Nodes to test per step (default: 5)
- `SEED`: Random seed (default: 0)
- `N_EVAL`: Number of evaluation episodes (default: 10)

### Environment Variables
- `AFEG_ENV_PATH`: Path to conda environment (default: `/n/netscratch/tambe_lab/Lab/msong300/.conda/envs/AFEG`)
- `DQN_QUICK_TEST`: Set to `1` for quick training (20 episodes), `0` for full (100 episodes)
- `GUROBI_LICENSE_SERVER`: Gurobi license server (required for compute nodes)

## Output

Results are saved to `results/{DISEASE}_T{THRESHOLD}_B{BUDGET}_seed{SEED}_{timestamp}/`:
- `summary.csv`: Algorithm performance summary
- `trajectories.csv`: Per-episode trajectory data
- `exp_curve_*.png`: Performance curve plots
