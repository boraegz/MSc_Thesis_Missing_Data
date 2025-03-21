from data_simulator import DataSimulator
import pandas as pd

def main():
    # Initialize the simulator
    simulator = DataSimulator(n_samples=1000, n_features=10, random_state=42)
    
    # Generate synthetic credit data
    print("1. Generating synthetic credit data...")
    data = simulator.simulate_credit_data()
     # Ensure all numeric values are positive (since they are about credit data)
    data.iloc[:, :] = data.abs()  # Convert all numeric values to their absolute values
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
    
    # Save test outputs for further use
    test_data_mcar = simulator.introduce_missingness(data.copy(), mechanism='MCAR', missing_proportion=0.2, missing_col='feature_0')
    test_data_mcar.to_csv("data_mcar.csv", index=False)

    test_data_mar = simulator.introduce_missingness(data.copy(), mechanism='MAR', missing_proportion=0.2, missing_col='feature_0')
    test_data_mar.to_csv("data_mar.csv", index=False)

    test_data_mnar = simulator.introduce_missingness(data.copy(), mechanism='MNAR', missing_proportion=0.2, missing_col='target')
    test_data_mnar.to_csv("data_mnar.csv", index=False)

    print("\nTest outputs saved as data_mcar.csv, data_mar.csv, and data_mnar.csv")

if __name__ == "__main__":
    main()