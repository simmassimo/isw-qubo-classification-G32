import pandas as pd
import numpy as np
from pathlib import Path


def feature_reduction(csv_path, output_path=None, sparsity_threshold=0.9):
    """
    Normalize a dataset by eliminating records with 90% or more zero elements.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file
    output_path : str, optional
        Path to save the reduced dataset. If None, returns DataFrame without saving
    sparsity_threshold : float, optional
        Threshold for zero element ratio (default: 0.9 = 90%)
    
    Returns:
    --------
    pd.DataFrame
        Dataset with sparse records removed
    """
    # Load the CSV file
    df = pd.read_csv(csv_path)
    
    # Calculate the number of features
    num_features = df.shape[1]
    
    # Function to check if a row has 90% or more zeros
    def is_sparse(row, threshold):
        zero_count = (row == 0).sum()
        zero_ratio = zero_count / len(row)
        return zero_ratio >= threshold
    
    # Filter out sparse records
    mask = ~df.apply(lambda row: is_sparse(row, sparsity_threshold), axis=1)
    reduced_df = df[mask].reset_index(drop=True)
    
    # Print statistics
    original_count = len(df)
    reduced_count = len(reduced_df)
    removed_count = original_count - reduced_count
    
    print(f"Original records: {original_count}")
    print(f"Removed records: {removed_count}")
    print(f"Remaining records: {reduced_count}")
    print(f"Sparsity threshold: {sparsity_threshold * 100}%")
    
    # Save to output path if specified
    if output_path:
        reduced_df.to_csv(output_path, index=False)
        print(f"Reduced dataset saved to: {output_path}")
    
    return reduced_df


if __name__ == "__main__":
    # Example usage
    input_csv = "input_dataset.csv"
    output_csv = "reduced_dataset.csv"
    
    # Run feature reduction with 90% sparsity threshold
    result_df = feature_reduction(input_csv, output_csv, sparsity_threshold=0.9)
