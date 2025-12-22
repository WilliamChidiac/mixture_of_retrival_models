import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.MOE_router.router import MOERouter, train, test_model
from pathlib import Path
import pandas as pd
import argparse
from typing import Tuple, List, Union
import json
curr_dir = Path(__file__).parent
score_path = curr_dir / 'data' / 'scores' 

def clean_df(df: pd.DataFrame) -> 'pd.DataFrame':
    """Cleans the input DataFrame by removing rows with NaN values in 'sparse_score' or 'dense_score' columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing 'sparse_score' and 'dense_score' columns.

    Returns:
        pd.DataFrame: Cleaned DataFrame with NaN values removed.
    """
    df = df[(df['sparse'] != 0.0) | (df['dense'] != 0.0)]
    return df

def remove_duplicates(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Removes duplicate queries from the training DataFrame that are present in the testing DataFrame.

    Args:
        train_df (pd.DataFrame): Training DataFrame containing a 'query' column.
        test_df (pd.DataFrame): Testing DataFrame containing a 'query' column.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Cleaned training and testing DataFrames with duplicates removed.
    """
    test_queries = set(test_df['query'].unique())
    train_df = train_df[~train_df['query'].isin(test_queries)].reset_index(drop=True)
    return train_df, test_df

def load_data(paths: List[str]) -> pd.DataFrame:
    """Loads and concatenates multiple CSV files into a single DataFrame.

    Args:
        paths (List[str]): List of file paths to CSV files.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """
    dfs = []
    for p in paths:
        if p and p != "":
            dfs.append(pd.read_csv(p))
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def main(train_score_paths: List[str], test_score_paths: List[str], epochs: int, batch_size: int, learning_rate: float, score_norm: str, run:int=1, train_val_split: float = 0.8) -> None:
    """Trains and tests the MOE Router model.

    Args:
        train_score_paths (List[str]): List of paths to the training scores CSV files.
        test_score_paths (List[str]): List of paths to the testing scores CSV files.
        epochs (int): _number of training epochs.
        batch_size (int): _batch size for training and testing.
        learning_rate (float): _learning rate for the optimizer.
        run (int): _run identifier for saving the model.
        train_val_split (float): _train-validation split ratio if train and test paths are the same.
    """
    run_dir = curr_dir / "runs"
    if run <= 0:
        run = 1
        while (run_dir / f"run_{run}").exists():
            run += 1
    print(f"Saving run to: {run_dir / f'run_{run}'}")
    run_path = run_dir / f"run_{run}"
    
    # Load training data
    train_df = load_data(train_score_paths)
    
    # Check if test paths are provided and valid
    has_test_data = len(test_score_paths) > 0 and any(p != "" for p in test_score_paths)
    
    if has_test_data:
        test_df = load_data(test_score_paths)
        
        # If train and test paths are identical (user passed same list to both)
        if train_score_paths == test_score_paths:
            full_df = clean_df(train_df)
            train_df = full_df.sample(frac=train_val_split, random_state=42)
            test_df = full_df.drop(train_df.index).reset_index(drop=True)
            train_df = train_df.reset_index(drop=True)
        else:
            train_df = clean_df(train_df)
            test_df = clean_df(test_df)
            train_df, test_df = remove_duplicates(train_df, test_df)
            
            if train_df.empty:
                raise ValueError("Training data is empty after removing duplicates. Please check that train and test datasets are different.")
    else:
        # No test data provided, split train data
        full_df = clean_df(train_df)
        train_df = full_df.sample(frac=train_val_split, random_state=42)
        test_df = full_df.drop(train_df.index).reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)

    router = MOERouter()
    trained_router = train(router, train_df, batch_size=batch_size, lr=learning_rate, epochs=epochs, score_norm=score_norm)
    results, total_accuracy= test_model(trained_router, test_df, score_norm=score_norm, batch_size=batch_size)
    run_path.mkdir(parents=True, exist_ok=True)
    trained_router.save_model(run_path)
    results.to_csv(run_path / "test_results.csv", index=False)
    ## save config as json
    config = {
        "train_score_paths": train_score_paths,
        "test_score_paths": test_score_paths,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "run": run,
        "score_norm": score_norm,
        "total_accuracy_on_weights_prediction": total_accuracy,
        "train_val_split": train_val_split
    }
    with open(run_path / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Test MOE Router")
    parser.add_argument("--train_score_path", nargs='+', type=str, required=True, help="Path(s) to training scores CSV file(s)")
    parser.add_argument("--test_score_path", nargs='*', type=str, default=[], help="Path(s) to testing scores CSV file(s)")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--score_norm", type=str, default="l1", help="Score normalization method")
    parser.add_argument("--run", type=int, default=-1, help="Run identifier for saving the model")
    parser.add_argument("--train_val_split", type=float, default=0.8, help="Train-validation split ratio if train and test paths are the same")
    
    args = parser.parse_args()
    main(args.train_score_path, args.test_score_path, args.epochs, args.batch_size, args.learning_rate, args.score_norm, args.run, args.train_val_split)
    