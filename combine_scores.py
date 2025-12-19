from pathlib import Path
import pandas as pd

curr_dir = Path(__file__).parent
score_path = curr_dir / 'data' / 'scores'

def combine_scores(split: str):
    split_path = score_path / split
    if not split_path.exists():
        print(f"Dossier {split_path} n'existe pas. Rien à combiner.")
        return

    all_files = list(split_path.glob("*.csv"))  # prend tous les CSV du dossier

    if not all_files:
        print(f"Aucun CSV trouvé dans {split_path}")
        return

    dfs = []
    for f in all_files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Erreur lecture {f}: {e}")

    if not dfs:
        print(f"Aucun CSV valide à combiner pour {split}")
        return

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(split_path / "combined_scores.csv", index=False)
    print(f"{split} : CSV combiné créé avec {len(combined_df)} lignes")


if __name__ == "__main__":
    for split in ["train", "test"]:
        combine_scores(split)
