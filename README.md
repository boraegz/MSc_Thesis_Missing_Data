# MSc Thesis: Missing Data & Reject Inference in Credit Scoring

This repository contains the implementation for the MSc Thesis focusing on **Missing Data solutions** and **Reject Inference** techniques in Credit Scoring. It provides a modular framework for simulating credit data, injecting missingness (MCAR, MAR, MNAR), applying correction methods, and evaluating their effectiveness.

## 🚀 Project Overview

The goal is to evaluate how well different strategies recover the "Oracle" (Ground Truth) performance when a significant portion of the target labels (Default/Repay) is missing due to rejection.

### Key Components:
1.  **Data Simulation (`src/data`)**: Generates synthetic credit application data (Oracle) and injects missingness based on various mechanisms (MCAR, Latent-MNAR, Policy-based Rejection).
2.  **Imputation & Correction (`src/imputation`, `src/reject_inference`)**:
    *   **Baseline**: Mean, Mode, Zero Imputation.
    *   **Advanced**: MICE (Multiple Imputation by Chained Equations), MissForest.
    *   **Reject Inference**: Inverse Probability Weighting (IPW), Augmentation (Self-training).
3.  **Experimentation (`src/experiments.py`)**: A pipeline to orchestrate the generation -> corruption -> correction -> evaluation loop.

## 📂 Project Structure

```
├── configs/
│   └── experiment_config.yaml  # Configuration for Dataset and Mechanisms
├── notebooks/
│   ├── walkthrough.ipynb       # Interactive demo of the project steps
│   └── ...
├── src/
│   ├── data/                   # Data Generation & Missingness Logic
│   ├── imputation/             # Imputation Classes (MICE, MissForest)
│   ├── reject_inference/       # Reject Inference Classes (IPW, Augmentation)
│   ├── evaluation.py           # Metric calculations (AUC, Brier)
│   ├── experiments.py          # Pipeline orchestration
│   └── utils.py                # Helper functions (Seeding, Config loading)
├── tests/                      # Unit tests for all modules
├── run_experiments.py          # Main execution script
└── environment.yaml            # Conda environment definition
```

## 🛠 Installation

This project uses **Micromamba** (or Conda) for dependency management.

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/boraegz/MSc_Thesis_Missing_Data.git
    cd MSc_Thesis_Missing_Data
    ```

2.  **Create the environment**:
    ```bash
    micromamba create -f environment.yaml
    micromamba activate thesis_env
    ```

3.  **Verify the setup**:
    ```bash
    python verify_env.py
    ```

## 🏃 Usage

### 1. Interactive Walkthrough
Open `notebooks/walkthrough.ipynb` in VSCode to interactively run the Data Generation and Imputation steps and visualize the output.

### 2. Run Experiments
To run the full suite of experiments (looping through MCAR/MNAR mechanisms and different missing rates):

```bash
python run_experiments.py
```

This will save the results to `results/experiment_results.csv`.

### 3. Run Unit Tests
To ensure everything is working correctly:

```bash
pytest tests/
```

## 🔬 Methodology

| Component | Methods Implemented |
|-----------|---------------------|
| **Mechanisms** | MCAR, MAR, MNAR (Latent), MNAR (Rejection Policy) |
| **Imputation** | Mean, Mode, Zero, MICE, MissForest |
| **Reject Inference** | Reweighting (IPW), Augmentation (Parceling) |
| **Evaluation** | AUC-ROC, Brier Score, Accuracy, F1 |

## 👤 Author
**Bora Eguz** (boraegz@gmail.com)
