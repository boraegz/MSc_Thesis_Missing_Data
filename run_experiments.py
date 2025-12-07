from src.utils import load_config
from src.experiments import ExperimentRunner
import pandas as pd
import os

def main():
    print("🚀 Starting Thesis Experiments...")
    
    # 1. Load Config
    config = load_config("configs/experiment_config.yaml")
    
    # 2. Initialize Runner
    runner = ExperimentRunner(config)
    
    # 3. Run Experiments
    # This loops over MCAR, MNAR and rates [0.1, 0.3, 0.5]
    # And methods: Oracle, Complete Case, Zero, Reweighting, Augmentation
    df_results = runner.run_all_experiments()
    
    # 4. Save Results
    output_dir = "results"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, "experiment_results.csv")
    df_results.to_csv(output_path, index=False)
    
    print(f"\n✅ Experiments Completed!")
    print(f"Results saved to: {output_path}")
    print(f"Total Runs: {len(df_results)}")
    print("\nPreview:")
    print(df_results[['mechanism', 'missing_rate', 'method', 'auc_roc', 'brier_score']])

if __name__ == "__main__":
    main()
