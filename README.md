# Credit Scoring with Missing Data

## Overview
This repository contains code and experiments for an MSc thesis on handling missing data in credit scoring, focusing on MCAR, MAR, and MNAR mechanisms.

## Repository Structure
- `data/`: Contains raw real dataset (not being used)
- `notebooks/`: Jupyter notebooks for experiments
  - `data_exploration.ipynb`: Exploratory data analysis
  - `missing_data_experiments.ipynb`: Main experiments with different missing data handling methods
- `src/`: Source code
  - `missing_data_handler.py`: Implementation of missing data handling methods
  - `model.py`: Credit scoring model implementation
  - `evaluation.py`: Model evaluation metrics
  - `utils.py`: Utility functions for visualization and data processing
  - `data_simulator.py`: Data generation and missingness introduction
- `requirements.txt`: Python dependencies

## Setup
1. Clone the repository:
   ```bash
   git clone boraegz/MSc_Thesis_Missing_Data
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Run notebooks in order:
   - First run `data_exploration.ipynb` to understand the data
   - Then run `missing_data_experiments.ipynb` for the main experiments

## Key Components
- **MissingDataHandler**: Implements various missing data handling methods:
  - Mean/Median imputation
  - Heckman correction
  - BASL (Bias-Aware Self-Learning)
- **CreditScoringModel**: Random Forest-based credit scoring model
- **ModelEvaluator**: Comprehensive model evaluation metrics
- **DataSimulator**: Generates synthetic credit data with controlled missingness

## Experiments
1. Data Exploration:
   - Analyze missingness patterns
   - Visualize feature distributions
   - Examine correlations

2. Missing Data Handling:
   - Compare different imputation methods
   - Evaluate impact on model performance
   - Analyze bias in predictions

3. Model Evaluation:
   - Performance metrics across methods
   - Feature importance analysis
   - Bias detection and mitigation
