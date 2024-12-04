# Credit Scoring with Missing Data

## Overview
This repository contains code and experiments for an MSc thesis on handling missing data in credit scoring, focusing on MCAR, MAR, and MNAR mechanisms.

## Repository Structure
- `data/`: Placeholder for raw and processed datasets.
- `notebooks/`: Jupyter notebooks demonstrating the workflow.
- `src/`: Source code for all classes and utility functions.
- `tests/`: Unit tests for validating the code.
- `requirements.txt`: Python dependencies.
- `README.md`: Instructions for usage.

## Setup
1. Clone the repository:
   ```bash
   git clone boraegz/MSc_Thesis_Missing_Data
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
- Run the Jupyter Notebook in the `notebooks/` folder for demonstrations.

## Classes and Methods
- **DataSimulator**: Generates synthetic data and introduces missingness.
- **MissingDataHandler**: Handles missing data using imputation, Heckman correction, and BASL.
- **ModelTrainer**: Trains machine learning models.
- **Evaluation**: Evaluates model performance using various metrics.

## Experiments
- Simulate a synthetic credit dataset.
- Introduce missingness using various mechanisms.
- Visualize missingness patterns.
- Handle missing data using different methods.
- Train and evaluate models.
- Compare performance of different techniques.
