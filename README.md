# Mixture of Retrieval Models (MOE)

This project implements an adaptive retrieval framework inspired by the Mixture of Experts (MOE) architecture. It combines the strengths of Sparse (BM25) and Dense (Semantic Search) retrieval models using a neural router that dynamically assigns weights to each expert based on the input query.

## Pipeline

The project workflow consists of three main steps: generating the dataset, training the router, and evaluating the complete model.

### 1. Generate Dataset

The first step is to generate the training data for the router. This involves running the individual experts (Sparse and Dense) on the target datasets to calculate their performance scores for each query. These scores serve as the "ground truth" for training the router.

**Script:** `generate_dataset.py`

**Usage:**
The script is currently configured to run on specific datasets defined in the `data_sets` list within the file (default is `['scifact']`).

```bash
python generate_dataset.py
```

This will generate CSV files in `data/scores/` containing the performance scores of each expert for the queries in the dataset.

### 2. Train Router

Once the scores are generated, the next step is to train the neural router. The router learns to predict the optimal weights for the Sparse and Dense experts based on the query embedding.

**Script:** `train_router.py`

**Arguments:**
- `--train_score_path`: Path(s) to the training scores CSV file(s) (Required).
- `--test_score_path`: Path(s) to the testing scores CSV file(s) (Optional).
- `--epochs`: Number of training epochs (Default: 5).
- `--batch_size`: Batch size (Default: 32).
- `--learning_rate`: Learning rate (Default: 1e-3).
- `--run`: Run identifier for saving the model (Default: auto-increment).

**Example:**
```bash
python train_router.py --train_score_path data/scores/train/scifact_topk25_scores.csv --test_score_path data/scores/test/scifact_topk25_scores.csv --epochs 10 --run 1
```

The trained model and configuration will be saved in `runs/run_<id>/`.

### 3. Use/Test Implementation

Finally, the trained router is used to perform retrieval on the test set. The `complete_model.py` script loads the trained router, runs the experts, and merges their results using the predicted weights. It then benchmarks the performance against the individual experts.

**Script:** `complete_model.py`

**Arguments:**
- `--corpus_names`: Name(s) of the corpus to evaluate (Required).
- `--run_id`: The run identifier of the trained router to load (Required).
- `--data_path`: Path to the data directory (Default: `src/data`).
- `--run_path`: Path to the runs directory (Default: `runs`).

**Example:**
```bash
python complete_model.py --corpus_names scifact --run_id 1
```

This will output the benchmark results (Precision@K, Recall@K) for the Sparse Expert, Dense Expert, and the MOE Model, and save them to `runs/run_<id>/benchmark_results.json`.

## Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```
