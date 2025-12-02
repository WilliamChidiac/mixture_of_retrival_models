import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.MOE_router.router import MOERouter, train, test_model
from pathlib import Path
import pandas as pd
import argparse
from typing import Tuple
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



def main(train_score_path: str, test_score_path: str, epochs: int, batch_size: int, learning_rate: float, run:int, score_norm: str) -> None:
    """Trains and tests the MOE Router model.

    Args:
        train_score_path (str): _path to the training scores CSV file.
        test_score_path (str): _path to the testing scores CSV file.
        epochs (int): _number of training epochs.
        batch_size (int): _batch size for training and testing.
        learning_rate (float): _learning rate for the optimizer.
        run (int): _run identifier for saving the model.
    """
    run_path = curr_dir / "run" / f"run_{run}"
    run_path.mkdir(parents=True, exist_ok=True)
    train_df = pd.read_csv(train_score_path)
    test_df = pd.read_csv(test_score_path)
    train_df = clean_df(train_df)
    test_df = clean_df(test_df)
    train_df, test_df = remove_duplicates(train_df, test_df)
    router = MOERouter()
    trained_router = train(router, train_df, batch_size=batch_size, lr=learning_rate, epochs=epochs, score_norm=score_norm)
    results = test_model(trained_router, test_df, score_norm=score_norm, batch_size=batch_size)
    trained_router.save_model(run_path)
    results.to_csv(run_path / "test_results.csv", index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and Test MOE Router")
    parser.add_argument("--train_score_path", type=str, default=score_path / "train" / "combined_scores.csv", help="Path to training scores CSV file")
    parser.add_argument("--test_score_path", type=str, default=score_path / "test" / "combined_scores.csv", help="Path to testing scores CSV file")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training and testing")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--score_norm", type=str, default="l1", help="Score normalization method")
    parser.add_argument("--run", type=int, default=1, help="Run identifier for saving the model")
    
    args = parser.parse_args()
    main(args.train_score_path, args.test_score_path, args.epochs, args.batch_size, args.learning_rate, args.run, args.score_norm)
    