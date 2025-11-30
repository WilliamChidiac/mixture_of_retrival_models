from pathlib import Path
from ..IR_models.sparse_model.sparse import BM25Expert
from ..IR_models.dense_model.dense import DenseExpert
from ..IR_models.models_api import IRExpert
from ..collections.data_utils import DataHandler
import numpy as np
from typing import Dict, List, Union, Optional, Tuple
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

class ExpertsResult:
    """
    A class to hold the results of the experts in an optimized structure for scoring.
    """
    def __init__(self, expert_names: List[str], queries_rank_matrices: List[np.ndarray], queries_doc_ids: List[List[str]]):
        """
        Initialize the ExpertsResult.

        Args:
            expert_names (List[str]): List of expert names (rows of the matrices).
            queries_rank_matrices (List[np.ndarray]): List of numpy arrays. Each array is (num_experts, num_unique_docs_for_query).
                                                      Values are ranks (1-based). 0 means not retrieved.
            queries_doc_ids (List[List[str]]): List of lists of doc_ids. queries_doc_ids[q] corresponds to columns of queries_rank_matrices[q].
        """
        self.expert_names = expert_names
        self.rank_matrices = queries_rank_matrices
        self.doc_ids = queries_doc_ids

    def compute_score(self, ground_truth: List[Dict[str, int]], k: int) -> Tuple[Dict[int, np.ndarray], List[str]]:
        """
        Compute the score for each expert based on the provided formula.
        
        Args:
            ground_truth (List[Dict[str, int]]): List of ground truth dictionaries (one per query).
                                                 Each dictionary maps doc_id to relevance score.
            k (int): The top K documents to consider.
            
        Returns:
            Tuple[Dict[int, np.ndarray], List[str]]: 
                Item 1: Dictionary mapping query index to an array of scores (one per expert).
                Item 2: List of expert names corresponding to the score array indices.
        """
        num_queries = len(self.rank_matrices)
        query_scores = {}
        
        for q_idx in range(num_queries):
            ranks = self.rank_matrices[q_idx] # (num_experts, num_docs)
            doc_ids = self.doc_ids[q_idx]
            gt = ground_truth[q_idx] if q_idx < len(ground_truth) else {}
            
            # Filter by top K
            # We only care about ranks <= k
            # Create a mask for retrieved in top k
            retrieved_mask = (ranks > 0) & (ranks <= k)
            
            # Count number of experts that retrieved each doc in top k
            # Shape: (num_docs,)
            doc_counts = np.sum(retrieved_mask, axis=0)
            
            # Avoid division by zero (if doc not retrieved by anyone in top k, count is 0, but we multiply by mask later so it's fine)
            # We can set 0 counts to 1 to avoid warning, as they will be masked out
            safe_counts = doc_counts.copy()
            safe_counts[safe_counts == 0] = 1
            inv_counts = 1.0 / safe_counts
            
            # Relevance vector
            # Shape: (num_docs,)
            relevance = np.array([gt.get(doc_id, 0) for doc_id in doc_ids])
            
            # Rank weights
            # Formula says "rank_K". Assuming 1/rank for now as it makes sense for "higher is better".
            rank_weights = np.zeros_like(ranks, dtype=np.float64)
            valid_ranks = ranks > 0
            rank_weights[valid_ranks] = 1.0 / ranks[valid_ranks]
            
            # Compute score per expert
            # Score = sum( rank_weight * relevance * inv_count )
            # We need to broadcast relevance and inv_count to (num_experts, num_docs)
            
            # Term 1: rank_weights (num_experts, num_docs)
            # Term 2: relevance (num_docs,) -> broadcast
            # Term 3: inv_counts (num_docs,) -> broadcast
            
            # Apply mask: only consider docs retrieved in top k
            term_matrix = rank_weights * relevance[None, :] * inv_counts[None, :] * retrieved_mask
            
            # Sum over docs
            expert_scores = np.sum(term_matrix, axis=1) # (num_experts,)
            
            query_scores[q_idx] = expert_scores
                
        return query_scores, self.expert_names

class Experts:
    """
    A class to manage multiple retrieval experts.
    """
    def __init__(self, **name_to_expert: Dict[str, IRExpert]):
        self.experts : Dict[str, IRExpert] = name_to_expert
        self.expert_names : List[str] = list(name_to_expert.keys())
        self.datahandler = None
        
    def build_indices(self, corpus_name:str, corpus: Dict[str, Dict[str, str]] = None, force: bool = False, save: bool = True) -> None:
        """
        Build indices for all experts.

        Args:
            corpus_name (str): Name of the corpus/dataset.
            corpus (Dict[str, Dict[str, str]], optional): Dictionary mapping doc_id to document content. 
                                                        If None, will load using DataHandler and the corpus name provided. Defaults to None.
            force (bool): Whether to force rebuild the indices even if they exist. Defaults to False.
        """
        if corpus is None and self.datahandler:
            self.datahandler = DataHandler(corpus_name)
        experts_items = list(self.experts.items())
        if not experts_items:
            return
        for name, expert in experts_items:
            expert.build_index(corpus_name, corpus=corpus, force=force, save=save)
            
            
            
    def search_query(self, query: Union[str, List[str]], top_k: int = 100, batch_size: int = 100) -> ExpertsResult:
        """
        Search the query using all experts.

        Args:
            query (Union[str, List[str]]): The input query string or list of query strings.
            top_k (int): Number of top documents to retrieve from each expert. Defaults to 100.
            batch_size (int): Number of queries to process at once. Defaults to 100.

        Returns:
            ExpertsResult: The combined search results from all experts.
        """
        if isinstance(query, str):
            queries = [query]
        else:
            queries = query
            
        # Perform search for all experts
        # results: expert_name -> list of list of doc_ids
        raw_results = {name: [] for name in self.experts}
        
        num_queries = len(queries)
        
        # Batch processing with tqdm
        for name, expert in self.experts.items():
            print(f"Searching with expert: {name}")
            for i in tqdm(range(0, num_queries, batch_size), desc="Searching queries"):
                batch_queries = queries[i : i + batch_size]
                batch_res = expert.search(batch_queries, top_k=top_k)
                raw_results[name].extend(batch_res)
            
        queries_rank_matrices = []
        queries_doc_ids = []
        
        for q_idx in range(num_queries):
            # 1. Collect all unique docs for this query
            unique_docs = set()
            for name in self.expert_names:
                unique_docs.update(raw_results[name][q_idx])
            
            sorted_unique_docs = sorted(list(unique_docs))
            doc_to_idx = {doc: i for i, doc in enumerate(sorted_unique_docs)}
            num_docs = len(sorted_unique_docs)
            
            # 2. Build rank matrix
            # Shape: (num_experts, num_docs)
            mat = np.zeros((len(self.expert_names), num_docs), dtype=np.int32)
            
            for i, name in enumerate(self.expert_names):
                retrieved = raw_results[name][q_idx] # List of doc_ids
                for rank_idx, doc_id in enumerate(retrieved):
                    if doc_id in doc_to_idx:
                        col = doc_to_idx[doc_id]
                        mat[i, col] = rank_idx + 1 # 1-based rank
            
            queries_rank_matrices.append(mat)
            queries_doc_ids.append(sorted_unique_docs)
            
        return ExpertsResult(self.expert_names, queries_rank_matrices, queries_doc_ids)
    
    def scoring_function(self, queries: Union[str, List[str]], ground_truth: List[Dict[str, int]], top_k: int = 100, batch_size: int = 100) -> Tuple[Dict[int, np.ndarray], List[str]]:
        """
        Get the scoring function outputs from all experts for the given queries.

        Args:
            queries (Union[str, List[str]]): The input query string or list of query strings.
            ground_truth (List[Dict[str, int]]): List of ground truth dictionaries (one per query).
            top_k (int): Top K documents to consider.
            batch_size (int): Batch size for search.
        Returns:
            Tuple[Dict[int, np.ndarray], List[str]]: 
                Item 1: Dictionary mapping query index to an array of scores (one per expert).
                Item 2: List of expert names corresponding to the score array indices.
        """
        results = self.search_query(queries, top_k=top_k, batch_size=batch_size)
        return results.compute_score(ground_truth, k=top_k)
    
    def run_pipeline(self, corpus_name: str, top_k: int, save_path: Path, batch_size: int = 100) -> None:
        """generates the score dataframes for a give corpus. The (query, scores) pair can be used to train the router model.

        Args:
            corpus_name (str): name of the corpus
            top_k (int): number of top documents to consider
            save_path (Path): location to save the datasets
            batch_size (int): number of queries to process at once to avoid OOM.
        """
        softmax = lambda x: np.exp(x) / np.sum(np.exp(x)) if np.sum(np.exp(x)) > 0 else x
        l1_normalization = lambda x: x / np.sum(x) if np.sum(x) > 0 else x
        if self.datahandler is None:
            self.datahandler = DataHandler(corpus_name)
            
        # Load raw data
        raw_queries = self.datahandler.load_queries()
        ground_truth = self.datahandler.load_qrels()
        
        # Prepare lists aligned by query ID
        query_ids = list(raw_queries.keys())
        queries_text = [raw_queries[qid]["text"] for qid in query_ids]
        gt_list = [ground_truth.get(qid, {}) for qid in query_ids]
        
        self.build_indices(corpus_name, force=False, save=True)
        
        results, ret_expert_names = self.scoring_function(queries=queries_text, ground_truth=gt_list, top_k=top_k, batch_size=batch_size)
        
        df = pd.DataFrame.from_dict(results, orient='index', columns=ret_expert_names)
        # Normalize scores
        nomrlized_columns = []
        for expert in ret_expert_names:
            for norm in ['l1', 'softmax']:
                nomrlized_columns.append(f"{expert}_{norm}")
        normilized_scores = {}
        for q_idx, scores in results.items():
            # Choose normalization method
            l1_scores = l1_normalization(scores)
            softmax_scores = softmax(scores)
            new_scores = l1_scores.tolist() + softmax_scores.tolist()
            normilized_scores[q_idx] = new_scores
        # add columns
        df_norm = pd.DataFrame.from_dict(normilized_scores, orient='index', columns=nomrlized_columns)
        total_df = pd.concat([df, df_norm], axis=1)

        # add query text
        query_texts_map = {idx: queries_text[idx] for idx in range(len(queries_text))}
        total_df.insert(0, 'query', pd.Series(query_texts_map))
        # Save to file
        save_path = save_path / f"{corpus_name}_topk{top_k}_scores.csv"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        total_df.to_csv(save_path, index_label='query_idx')






