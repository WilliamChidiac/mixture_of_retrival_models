import os 
import sys
from pathlib import Path
curr_dir = Path(__file__).parent
sys.path.append(str(curr_dir))
from typing import List, Dict, Union, Tuple
import numpy as np
import pandas as pd

from src.IR_models.dense_model import dense as dm
from src.IR_models.sparse_model import sparse as sm
from src.MOE_router import experts as ex
from src.MOE_router import router as rt 
from src.collections import DataHandler
from train_router import train

curr_path = Path(__file__).parent
_data_path = curr_path / 'data'
class Model:
    def __init__(self, router : rt.MOERouter, experts: ex.Experts, data_path: Path = _data_path, corpus_name: str = "fiqa", top_k: int = 25):
        """ Initializes the MOE Model with a router and expert models.

        Args:
            router (rt.MOERouter): _the MOE router model. 
            experts (ex.Experts): _the expert models.
            data_path (Path, optional): _path to the data directory. Defaults to _data_path.
            corpus_name (str, optional): _name of the corpus. Defaults to "fiqa".
            top_k (int, optional): _number of top results to consider. Defaults to 25.
        """
        
        self.router : rt.MOERouter = router
        self.experts : ex.Experts = experts
        self.top_k = top_k
        self.corpus_name = corpus_name
        self.data_path = data_path
        dh = DataHandler(dataset_name= corpus_name, root_data_dir=data_path)
        qrels = dh.load_qrels()
        queries = dh.load_queries()
        self.queries = {}
        self.qrels = {}
        for qid, rel_doc in qrels.items():
            query_text = queries.get(qid, {}).get("text", "")
            self.queries[qid] = query_text
            self.qrels[query_text] = rel_doc        
    
    def load_router(self, path: Path) -> None:
        """Loads the router model from the specified path."""
        if path.exists():
            self.router.load_model(path)
            self.router.is_trained = True
            print(f"Router chargé depuis {path}")
        else:
            self.router.is_trained = False
            print(f"Aucun router trouvé à {path}, fusion simple sera utilisée")


    
    def load_experts(self) -> None:
        """Loads all expert models."""
        self.experts.build_indices(self.corpus_name, force=False)
        
    def get_res_per_expert(self, query: List[str]) -> Dict[str, List[List[str]]]:
        """Gets results from each expert model for the given queries.

        Args:
            query (List[str]): List of query strings.
        Returns:
            Dict[str, List[List[str]]]: Dictionary mapping each expert to its results for the queries
        """
        expert_results = {}
        for expert_name, expert_model in self.experts.experts.items():
            expert_results[expert_name] = expert_model.search(query, top_k=self.top_k)
        return expert_results
    
    def merge_results(self, exp_res: Dict[str, List[List[str]]], weights: Dict[str, List[float]], queries: List[str]) -> Dict[str, List[str]]:
        """Merges results from multiple experts based on the provided weights.

        Args:
            exp_res (Dict[str, List[List[str]]]): Results from each expert model.
            weights (Dict[str, List[float]]): Weights for each expert model per query.
            queries (List[str]): List of query strings.
        Returns:
            Dict[str, List[str]]: Merged results for each query.
        """
        final_results = {}
        for i, query in enumerate(queries):
            doc_scores : Dict[str, float] = {}
            for expert_name, expert_results in exp_res.items():
                for rank, doc in enumerate(expert_results[i]):
                    score = weights[expert_name][i] / (rank + 1)  # Simple scoring mechanism
                    if doc in doc_scores:
                        doc_scores[doc] += score
                    else:
                        doc_scores[doc] = score
            # Sort and get top_k results
            sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
            final_results[query] = [doc for doc, score in sorted_docs[:self.top_k]]
        return final_results
    
    def search(self, exp_res: Dict[str, List[List[str]]] = None) -> Dict[str, List[str]]:
        query = list(self.queries.values())
        
        
        # ✅ Vérifier si le router est entraîné
        if hasattr(self.router, "get_weights") and getattr(self.router, "is_trained", False):
            # Poids appris pour chaque requête
            weights = self.router.get_weights(query)
            weights_dict = {expert: weights[:, idx].tolist() for idx, expert in enumerate(self.experts.experts.keys())}
            print("Router entraîné : utilisation des poids appris")
        else:
            # Fusion simple si router non entraîné
            weights_dict = {expert: [1.0]*len(query) for expert in self.experts.experts.keys()}
            print("Router non entraîné : utilisation de fusion simple")
        
        if exp_res is None:
             exp_res = self.get_res_per_expert(query)
        
        final_results = self.merge_results(exp_res, weights_dict, query)
        return final_results

    def compute_score_per_query(self, res:Union[Dict[str, List[str]], List[List[str]]]) -> Dict[str, Dict[str, float]]:
        """Computes evaluation scores for each query based on the results and ground truth.

        Args:
            res (Dict[str, List[str]]): Retrieved results for each query.
        Returns:
            Dict[str, Dict[str, float]]: Evaluation scores for each query.
        """
        scores = {}
        if isinstance(res, list):
            res_dict = {}
            for idx, query in enumerate(self.queries.values()):
                res_dict[query] = res[idx]
            res = res_dict
        for query, retrieved_docs in res.items():
            relevant_docs = self.qrels.get(query, {})
            scores[query] = {} 
            for k in [1, 5, 10, 25]:
                retrieved_at_k = retrieved_docs[:k]
                relevance_retrieved = sum(1*relevant_docs[doc_id] for doc_id in retrieved_at_k if doc_id in relevant_docs)
                precision = relevance_retrieved / k
                recall = relevance_retrieved / len(relevant_docs) if len(relevant_docs) > 0 else 0.0
                scores[query][f'precision@{k}'] = precision
                scores[query][f'recall@{k}'] = recall
        return scores
    
    def benchmark_results(self) -> Dict[str, float]:
        """Benchmarks the MOE model's results against ground truth and compares its performance with individual experts.

        Args:
            queries (List[str]): List of query strings.
            ground_truth (Dict[str, List[str]]): Ground truth results for the queries.
        Returns:
            Dict[str, float]: Benchmarking metrics (e.g., precision, recall).
        """
        res_per_exp = self.get_res_per_expert(list(self.queries.values()))
        moe_results = self.search(exp_res=res_per_exp)
        res_per_exp['MOE'] = moe_results
        all_scores = {}
        for model_name, results in res_per_exp.items():
            scores = self.compute_score_per_query(results)
            avg_scores = {}
            for k in [1, 5, 10, 25]:
                precision_key = f'precision@{k}'
                recall_key = f'recall@{k}'
                avg_precision = np.mean([scores[q][precision_key] for q in scores])
                avg_recall = np.mean([scores[q][recall_key] for q in scores])
                avg_scores[precision_key] = avg_precision
                avg_scores[recall_key] = avg_recall
            all_scores[model_name] = avg_scores
        return all_scores
    
def main():
    ir_models = {
        "sparse": sm.BM25Expert(data_path=_data_path),
        "dense": dm.DenseExpert(data_path=_data_path)
    }
    experts = ex.Experts(**ir_models)
    router = rt.MOERouter()

    datasets = ["scifact", "fiqa"]  # datasets exploitables

    for ds in datasets:
        print(f"\n=== Benchmark on {ds} ===")
        model = Model(router, experts, corpus_name=ds)

        # --- Chargement du router entraîné si disponible ---
        router_path = curr_dir / "run" / "run_1"  # chemin du router entraîné
        if router_path.exists():
            model.load_router(router_path)
            print(f"Router entraîné chargé depuis {router_path}")
        else:
            print("Aucun router entraîné trouvé, fusion simple utilisée.")

        # --- Charger les experts ---
        model.load_experts()

        # --- Lancer le benchmark ---
        benchmark_scores = model.benchmark_results()

        # --- Affichage des scores ---
        import json
        print(json.dumps(benchmark_scores, indent=4))

if __name__ == "__main__":
    main()


    