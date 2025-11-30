import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.MOE_router import experts
from src.IR_models.dense_model import dense
from src.IR_models.sparse_model import sparse
from pathlib import Path

data_sets = ["trec-covid", "quora", "scidocs", "fiqa", "msmarco", "nq"]
file_path = Path(__file__).parent
def _run_dataset(ds):

    exps = {
        "sparse": sparse.BM25Expert(data_path=file_path / "data"),
        "dense": dense.DenseExpert(data_path=file_path / "data")
    }
    moe = experts.Experts(**exps)
    moe.run_pipeline(corpus_name=ds, top_k=25, save_path=Path("./test_outputs_scores"))
    return ds

if __name__ == "__main__":
    for ds in data_sets:
        try:
            _run_dataset(ds)
        except Exception as e:
            print(f"Dataset {ds} failed: {e}")
