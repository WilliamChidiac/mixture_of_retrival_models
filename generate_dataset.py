import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.MOE_router import experts
from src.IR_models.dense_model import dense
from src.IR_models.sparse_model import sparse
from pathlib import Path

data_sets = ["trec-covid", "quora", "scidocs", "fiqa", "msmarco", "nq"]
try:
    file_path = Path(__file__).parent
except:
    file_path = Path(".")
data_path = file_path / "data"
save_path = data_path / "scores"
def _run_dataset(ds):

    exps = {
        "sparse": sparse.BM25Expert(data_path=data_path),
        "dense": dense.DenseExpert(data_path=data_path)
    }
    moe = experts.Experts(**exps)
    moe.run_pipeline(corpus_name=ds, top_k=25, save_path=save_path)
    return ds

if __name__ == "__main__":
    for ds in data_sets:
        try:
            _run_dataset(ds)
        except Exception as e:
            print(f"Dataset {ds} failed: {e}")
