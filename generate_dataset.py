import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from src.MOE_router import experts
from src.IR_models.dense_model import dense
from src.IR_models.sparse_model import sparse
from pathlib import Path

# Datasets choisis
data_sets = ["scifact","fiqa"]

# Splits disponibles pour chaque dataset
available_splits = {
    "scifact": ["train", "test"],
    "fiqa": ["train", "test"]
}   

try:
    file_path = Path(__file__).parent
except:
    file_path = Path(".")
data_path = file_path / "data"
save_path = data_path / "scores"

def _run_dataset(ds, split):
    """Exécute le pipeline pour un dataset et un split donné."""
    exps = {
        "sparse": sparse.BM25Expert(data_path=data_path),
        "dense": dense.DenseExpert(data_path=data_path)
    }
    moe = experts.Experts(**exps)
    sp = save_path / split
    sp.mkdir(parents=True, exist_ok=True)  # Assure que le dossier existe
    moe.run_pipeline(corpus_name=ds, top_k=25, save_path=sp, dataset=split)
    return ds

if __name__ == "__main__":
    # Génération des scores bruts pour tous les datasets
    for ds in data_sets:
        splits = available_splits.get(ds, ["test"])
        for split in splits:
            try:
                _run_dataset(ds, split)
                print(f"{ds} - {split} OK")
            except Exception as e:
                print(f"{ds} - {split} failed: {e}")

    # Combiner tous les CSV bruts pour créer train/test
    from combine_scores import combine_scores
    for split in ["train", "test"]:
        combine_scores(split)
