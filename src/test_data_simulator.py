from data_simulator import DataSimulator
import pandas as pd

def main():
    # Initialize the simulator
    simulator = DataSimulator(n_samples=1000, n_features=10, random_state=42)
    
    # Generate synthetic credit data
    print("1. Generating synthetic credit data...")
    data = simulator.simulate_credit_data()
    print("\nFirst few rows of the generated data:")
    print(data.head())
    print("\nData shape:", data.shape)
    print("\nMissing values before introducing missingness:")
    print(data.isnull().sum())
    
    # Test each missingness mechanism
    mechanisms = ['MCAR', 'MAR', 'MNAR']
    
    for mechanism in mechanisms:
        print(f"\n2. Testing {mechanism} missingness mechanism...")
        # Create a copy of the data for each test
        test_data = data.copy()
        
        # Introduce missingness
        test_data = simulator.introduce_missingness(
            test_data,
            mechanism=mechanism,
            missing_proportion=0.2,
            missing_col='feature_0'  # This is only used for MCAR and MAR
        )
        
        # Show results
        print(f"\nMissing values after introducing {mechanism}:")
        print(test_data.isnull().sum())
        
        # Display appropriate percentage based on mechanism
        if mechanism == 'MNAR':
            col = 'target'
        else:
            col = 'feature_0'
            
        missing_percentage = (test_data[col].isnull().sum() / len(test_data)) * 100
        print(f"\nPercentage of missing values in {col}: {missing_percentage:.2f}%")

if __name__ == "__main__":
    main() 