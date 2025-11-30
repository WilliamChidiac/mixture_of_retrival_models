from ..models_api import IRExpert
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import pickle
import tqdm
import torch
from typing import List, Dict, Any, Union
from pathlib import Path
from ...collections.data_utils import DataHandler

class DenseExpert(IRExpert):
    """
    Dense retrieval expert using Sentence Transformers and FAISS.
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', data_path: Path = None):
        """
        Initialize the DenseExpert.

        Args:
            model_name (str): Name of the sentence-transformer model. Defaults to 'all-MiniLM-L6-v2'.
        """
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.corpus_ids = []
        self.doc_id_to_index = {}
        self.data_path = data_path
        self.save_path = data_path / "indexed_data" / "dense" if data_path else None
        
    def build_index(self, corpus_name:str, corpus: Dict[str, Dict[str, str]] = None, force: bool = False, save: bool = True) -> None:
        """
        Build the FAISS index from the corpus.

        Args:
            corpus (Dict[str, Dict[str, str]]): Dictionary mapping doc_id to document content.
        """
        print("Building Dense index...")
        if corpus is None:
            index_corpus_path : Path= self.save_path/ corpus_name
            print(f"Checking for existing Dense index at {index_corpus_path}")
            print(f"Force rebuild: {force}")
            print(f"Index exists: {index_corpus_path.exists()}")
            if index_corpus_path.exists() and not force:
                print(f"Loading existing Dense index from {index_corpus_path}")
                self.load_index(index_corpus_path)
                return
            
            print(f"Loading corpus using DataHandler for corpus: {corpus_name}")
            downloader = DataHandler(corpus_name, root_data_dir=self.data_path)
            corpus = downloader.load_corpus()
        elif not isinstance(corpus, dict):
            raise ValueError("when provided, the corpus must be a Dict[doc_id, Dict[doc_component, component_data]] object ")
        
        self.corpus_ids = list(corpus.keys())
        self.doc_id_to_index = {doc_id: i for i, doc_id in enumerate(self.corpus_ids)}
        sentences = []
        for doc_id in tqdm.tqdm(self.corpus_ids):
            doc = corpus[doc_id]
            text = doc.get("title", "") + " " + doc.get("text", "")
            sentences.append(text)
            
        embeddings = self.model.encode_document(sentences, show_progress_bar=True, convert_to_numpy=True)
        
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension) # Inner Product (Cosine Similarity if normalized)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        print("Dense index built.")
        if save and self.save_path:
            self.save_index(path=self.save_path / corpus_name)

    def save_index(self, path: str) -> None:
        """
        Save the FAISS index to disk.

        Args:
            path (str): Directory path to save the index.
        """
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "corpus_ids.pkl"), "wb") as f:
            pickle.dump(self.corpus_ids, f)
        with open(os.path.join(path, "doc_id_to_index.pkl"), "wb") as f:
            pickle.dump(self.doc_id_to_index, f)
        print(f"Dense index saved to {path}")

    def load_index(self, path: str) -> None:
        """
        Load the FAISS index from disk.

        Args:
            path (str): Directory path to load the index from.
        """
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        with open(os.path.join(path, "corpus_ids.pkl"), "rb") as f:
            self.corpus_ids = pickle.load(f)
        if os.path.exists(os.path.join(path, "doc_id_to_index.pkl")):
             with open(os.path.join(path, "doc_id_to_index.pkl"), "rb") as f:
                self.doc_id_to_index = pickle.load(f)
        else:
             self.doc_id_to_index = {doc_id: i for i, doc_id in enumerate(self.corpus_ids)}
        print(f"Dense index loaded from {path}")

    def search(self, query: Union[str, List[str]], top_k: int = 10) -> List[List[str]]:
        """
        Search for the query in the FAISS index.

        Args:
            query (Union[str, List[str]]): Query string or list of query or queries strings.
            top_k (int): Number of top results to return. Defaults to 10.

        Returns:
            List[List[str]]: List of lists of top_k document IDs ranked by relevance for each query.
        """
        if isinstance(query, str):
            query = [query]
        query_embedding = self.model.encode_query(query, convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        _ , indices = self.index.search(query_embedding, top_k)
        top_k_doc_ids_per_query = []
        for doc_indices in indices:
            top_k_doc_ids = [self.corpus_ids[idx] for idx in doc_indices]
            top_k_doc_ids_per_query.append(top_k_doc_ids)
        return top_k_doc_ids_per_query 

    def score_pairs(self, query: str, doc_ids: List[str]) -> np.ndarray:
        """
        Score a list of documents for a given query.

        Args:
            query (str): Query string.
            doc_ids (List[str]): List of document IDs to score.

        Returns:
            np.ndarray: Array of scores corresponding to the doc_ids.
        """
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        scores = []
        for doc_id in doc_ids:
            if doc_id in self.doc_id_to_index:
                idx = self.doc_id_to_index[doc_id]
                # Reconstruct vector from FAISS (only works if index supports it, IndexFlatIP does)
                doc_embedding = self.index.reconstruct(idx)
                doc_embedding = doc_embedding.reshape(1, -1)
                # Dot product
                score = np.dot(query_embedding, doc_embedding.T).item()
                scores.append(score)
            else:
                scores.append(0.0)
        return np.array(scores)
