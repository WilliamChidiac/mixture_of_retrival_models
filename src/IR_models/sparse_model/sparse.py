from ..models_api import IRExpert
from rank_bm25 import BM25Okapi
import pickle
import os
import nltk
from nltk.tokenize import word_tokenize
import tqdm
import numpy as np
from typing import List, Dict, Any, Union
from pathlib import Path
from ...collections.data_utils import DataHandler

# Ensure nltk data is downloaded
nltk_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../.venv", "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', download_dir=nltk_data_path)

class BM25Expert(IRExpert):
    """
    Sparse retrieval expert using BM25.
    """
    def __init__(self, data_path: Path = None):
        """
        Initialize the BM25Expert.
        """
        self.bm25 = None
        self.corpus_ids = []
        self.doc_id_to_index = {}
        self.data_path = data_path
        self.save_path = data_path / "indexed_data" / "sparse" if data_path else None
        
    def tokenize(self, text: Union[str, List[str]]) -> List[List[str]]:
        """
        Tokenize the input text.

        Args:
            text (Union[str, List[str]]): Input text or list of texts.

        Returns:
            List[List[str]]: List of tokens for each input text.
        """
        if isinstance(text, list):
            return [word_tokenize(t.lower()) for t in text]
        else:
            return [word_tokenize(text.lower())]

    def build_index(self, corpus_name: str, corpus: Dict[str, Dict[str, str]] = None, force: bool = False, save: bool = True) -> None:
        """
        Build the BM25 index from the corpus.

        Args:
            corpus_name (str): Name of the corpus/dataset.
            corpus (Dict[str, Dict[str, str]], optional): Dictionary mapping doc_id to document content. 
                                                        If None, will load using DataHandler and the corpus name provided. Defaults to None.
            force (bool): Whether to force rebuild the index even if it exists. Defaults to False.
            save (bool): Whether to save the index after building. Defaults to True.
        """
        if corpus is None:
            index_corpus_path : Path= self.save_path/ corpus_name
            
            if index_corpus_path.exists() and not force:
                print(f"Loading existing BM25 index from {index_corpus_path}")
                self.load_index(index_corpus_path)
                return
        
            downloader = DataHandler(corpus_name, root_data_dir=self.data_path)
            corpus = downloader.load_corpus()
        elif not isinstance(corpus, dict):
            raise ValueError("when provided, the corpus must be a Dict[doc_id, Dict[doc_component, component_data]] object ")
        
        # corpus is dict: doc_id -> {title, text}
        print("Building BM25 index...")
        tokenized_corpus = []
        self.corpus_ids = list(corpus.keys())
        for doc_id in tqdm.tqdm(self.corpus_ids):
            doc = corpus[doc_id]
            text = doc.get("title", "") + " " + doc.get("text", "")
            tokenized_corpus.append(self.tokenize(text)[0])
            
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.doc_id_to_index = {doc_id: i for i, doc_id in enumerate(self.corpus_ids)}
        print("BM25 index built.")
        if save:
            self.save_index(path=self.save_path / corpus_name if self.save_path else None)

    def save_index(self, path: Path = None) -> None:
        """
        Save the BM25 index to disk.

        Args:
            path (Path): Directory path to save the index. If None, uses self.save_path.
        """
        if path is None:
            path = self.save_path
        if path is None:
            print("Skipping save: No path specified to save the index.")
            return
        os.makedirs(path, exist_ok=True)
        with open(path / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)
        with open(path / "corpus_ids.pkl", "wb") as f:
            pickle.dump(self.corpus_ids, f)
        with open(path / "doc_id_to_index.pkl", "wb") as f:
            pickle.dump(self.doc_id_to_index, f)
        print(f"BM25 index saved to {path}")

    def load_index(self, path: Path) -> None:
        """
        Load the BM25 index from disk.

        Args:
            path (Path): Directory path to load the index from.
        """
        with open(path / "bm25.pkl", "rb") as f:
            self.bm25 = pickle.load(f)
        with open(path / "corpus_ids.pkl", "rb") as f:
            self.corpus_ids = pickle.load(f)
        if (path / "doc_id_to_index.pkl").exists():
             with open(path / "doc_id_to_index.pkl", "rb") as f:
                self.doc_id_to_index = pickle.load(f)
        else:
             self.doc_id_to_index = {doc_id: i for i, doc_id in enumerate(self.corpus_ids)}
        print(f"BM25 index loaded from {path}")

    def search(self, query: Union[str, List[str]], top_k: int = 10) -> List[List[str]]:
        """
        Search for the query in the BM25 index.

        Args:
            query (Union[str, List[str]]): Query string or list of query strings.
            top_k (int): Number of top results to return. Defaults to 10.

        Returns:
            List[List[str]]: List of lists of top_k document IDs ranked by relevance for each query.
        """
        tokenized_query = self.tokenize(query)
        scores = np.array([self.bm25.get_scores(q) for q in tokenized_query])
        top_k_indices = np.argsort(scores, axis=1)[:, ::-1][:, :top_k]
        corpus_ids_array = np.array(self.corpus_ids)
        top_k_doc_ids = corpus_ids_array[top_k_indices]
        return top_k_doc_ids.tolist()

    def score_pairs(self, query: str, doc_ids: List[str]) -> np.ndarray:
        """
        Score a list of documents for a given query.

        Args:
            query (str): Query string.
            doc_ids (List[str]): List of document IDs to score.

        Returns:
            np.ndarray: Array of scores corresponding to the doc_ids.
        """
        tokenized_query = self.tokenize(query)
        # rank_bm25 doesn't support efficient batch scoring for specific docs easily without accessing internal structures
        # But we can get all scores and pick. Or calculate manually.
        # get_scores calculates for all.
        # If len(doc_ids) is small, maybe manual calculation is faster?
        # BM25Okapi: sum(IDF * ((f * (k1 + 1)) / (f + k1 * (1 - b + b * |D| / avgdl))))
        # It's easier to just use get_scores() which is vectorized, and select indices.
        
        all_scores = self.bm25.get_scores(tokenized_query)
        scores = []
        for doc_id in doc_ids:
            if doc_id in self.doc_id_to_index:
                idx = self.doc_id_to_index[doc_id]
                scores.append(all_scores[idx])
            else:
                scores.append(0.0)
        return np.array(scores)

if __name__ == "__main__":
    # Simple test
    corpus = {
        "doc1": {"title": "The cat in the hat", "text": "A story about a cat."},
        "doc2": {"title": "Green eggs and ham", "text": "A story about green eggs."},
        "doc3": {"title": "The quick brown fox", "text": "A story about a fox."}
    }
    
    bm25_expert = BM25Expert()
    bm25_expert.build_index(corpus)
    
    query = "cat story"
    results = bm25_expert.search(query, top_k=2)
    print("Search Results:", results)
    
    doc_ids = ["doc1", "doc2", "doc3"]
    scores = bm25_expert.score_pairs(query, doc_ids)
    print("Scores:", scores)