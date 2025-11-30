import os
import json
import csv
import shutil
from typing import Dict, Any, Optional, Tuple
from beir import util
from pathlib import Path

_file_path = Path(__file__).parent
_root_data_dir = _file_path / ".." / ".." / "data"

class Qrels:
    """
    Handles loading and storing of qrels (query relevance judgments).
    """
    def __init__(self, file: Path = None, delimiter: str = "\t"):
        """Initialize Qrels object.

        Args:
            file (Path, optional): Path to the qrels file (accepted formats: .json, .tsv, .csv). Defaults to None.
            delimiter (str, optional): Delimiter used in TSV/CSV files. Defaults to "\t".
        """
        self.qrels: Dict[str, Dict[str, int]] = {}
        if file is not None:
            if file.suffix == ".json":
                self.load_from_json(file)
            elif file.suffix in {".tsv", ".csv"}:
                self.load_from_tsv(file, delimiter=delimiter)
        print(f"Loaded Qrels with {len(self.qrels)} queries.")
    
    def load_from_json(self, json_file: Path) -> None:
        """Load qrels from a JSON file.

        Args:
            json_file (Path): Path to the JSON file containing qrels.
        """
        with open(json_file, 'r') as f:
            self.qrels = json.load(f)
            
    def load_from_tsv(self, tsv_file: Path, delimiter: str = "\t") -> None: 
        """Load qrels from a TSV or CSV file.

        Args:
            tsv_file (Path): Path to the TSV or CSV file containing qrels.
            delimiter (str, optional): Delimiter used in the file. Defaults to "\t".
        """
        def extract_from_line(line: str) -> Tuple[str, str, int]:
            query_id, doc_id, score = line.strip().split(delimiter)
            return str(query_id), str(doc_id), int(score)
        with open(tsv_file, 'r') as f:
            f.readline()  # skip header
            for line in f:
                query_id, doc_id, score = extract_from_line(line)
                if score != 0:
                    self.qrels.setdefault(query_id, {})[doc_id] = score
    
    def save_to_json(self, json_file: Path) -> None:
        """Save qrels to a JSON file.

        Args:
            json_file (Path): Path to the JSON file to save qrels.
        """
        with open(json_file, 'w') as f:
            json.dump(self.qrels, f)
        
    
    def get_judgment(self, q_id: str, doc_id: str) -> Optional[int]:
        """Get the relevance judgment for a given query and document.

        Args:
            q_id (str): Query ID.
            doc_id (str): Document ID.
        Raises:
            KeyError: If the query ID is not found in the qrels.

        Returns:
            Optional[int]: Relevance judgment score or 0 if not found.
        """
        try:
            dict_docs = self.qrels[q_id]
            return dict_docs.get(doc_id, 0)
        except KeyError:
            raise KeyError(f"Query ID {q_id} not found in qrels.")
    
        
class DataHandler:
    """
    Handles downloading, caching, and loading of BeIR datasets.
    """
    def __init__(self, dataset_name: str, root_data_dir: Path = _root_data_dir, temp_dir: str = "temp"):
        """
        Initialize the DataHandler.

        Args:
            dataset_name (str): Name of the BeIR dataset (e.g., 'scifact').
            root_data_dir (str): Root directory to store data. Defaults to "data".
        """
        self.temp_dir = root_data_dir / temp_dir
        self.dataset_name = dataset_name
        self.root_data_dir = root_data_dir
        self.raw_corpus_dir =root_data_dir / "raw_corpus" / dataset_name
        
        # Define paths according to user requirement
        self.documents_path = self.raw_corpus_dir / "documents.json" 
        self.queries_results_dir = self.raw_corpus_dir / "queries_results_id"
        self.queries_path = self.queries_results_dir / "queries.json"
        self.results_path = os.path.join(self.queries_results_dir, "results.json")
        
        self._download_and_cache()

    def _download_and_cache(self) -> None:
        """
        Downloads the dataset if not already present and caches it in the specified structure.
        """
        if os.path.exists(self.raw_corpus_dir) and os.path.exists(self.documents_path):
            print(f"Dataset {self.dataset_name} already exists in {self.raw_corpus_dir}")
            return

        print(f"Downloading {self.dataset_name}...")
        url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(self.dataset_name)
        temp_dir = self.temp_dir / self.dataset_name
        data_path = temp_dir / self.dataset_name
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
            data_path = Path(util.download_and_unzip(url, temp_dir))
        
        os.makedirs(self.raw_corpus_dir, exist_ok=True)
        os.makedirs(self.queries_results_dir, exist_ok=True)
        extraction = 0
        # 1. Documents
        src_corpus = data_path / "corpus.jsonl"
        if src_corpus.exists():
            corpus = {}
            with open(src_corpus, 'r') as f, open(self.documents_path, 'w') as dest_f:
                for line in f:
                    doc = json.loads(line)
                    corpus[doc.get('_id')] = doc
                json.dump(corpus, dest_f)
                extraction += 1
        
        # 2. Queries
        src_queries = data_path / "queries.jsonl"
        if src_queries.exists():
            with open(src_queries, 'r') as src_f, open(self.queries_path, 'w') as dest_f:
                dic = {}
                for line in src_f:
                    q : dict = json.loads(line)
                    dic[q.pop('_id')] = q
                json.dump(dic, dest_f)
                extraction += 1
                
        # 3. Results (Qrels)
        qrels_dir = data_path / "qrels"
        res_fil = "test.tsv"
        if (qrels_dir / res_fil).exists():
                qrels = Qrels(file=(qrels_dir / res_fil))
                qrels.save_to_json(Path(self.results_path))
                extraction += 1
            
        # Cleanup
        if extraction < 3:
            print(f"Warning: In dataset {self.dataset_name}, expected to extract 3 components but only extracted {extraction}.")
        else:
            shutil.rmtree(temp_dir)
            print(f"Dataset cached at {self.raw_corpus_dir}")

    def load_corpus(self, corpus_path:Path = None) -> Dict[str, Dict[str, Any]]:
        """
        Loads the corpus from the cached file.

        Args:
            corpus_path (Path, optional): Path to the corpus file.  If None, uses the default documents path.

        Returns:
            Dict[str, Dict[str, Any]]: A dictionary mapping document IDs to document content.
        """
        return json.load(open(corpus_path if corpus_path else self.documents_path, 'r'))

    def load_queries(self, queries_path:Path = None) -> Dict[str, str]:
        """
        Loads queries from the cached file.

        Args:
            queries_path (Path, optional): Path to the queries file. If None, uses the default queries path.

        Returns:
            Dict[str, str]: A dictionary mapping query IDs to query text.
        """
        return json.load(open(queries_path if queries_path else self.queries_path, 'r'))

    def load_qrels(self, qrels_path: Path = None, delimiter : str = '\t') -> Dict[str, Dict[str, int]]:
        """
        Loads qrels (relevance judgments) from the cached file.

        Args:
            split (Optional[str]): The split to load (e.g., 'train', 'test', 'dev'). 
                                   If None, loads all.
            qrels_path (Path, optional): Path to the qrels file. If None, uses the default results path.

        Returns:
            Dict[str, Dict[str, int]]: A dictionary mapping query IDs to a dictionary of 
                                       document IDs and their relevance scores.
        """
        return Qrels(file=(qrels_path if qrels_path else Path(self.results_path)), delimiter=delimiter).qrels

    @staticmethod
    def print_available_datasets() -> None:
        """
        Prints an overview of all available BeIR datasets.
        Note: BeIR does not provide a built-in function to list datasets with statistics programmatically.
        This method uses huggingface_hub to list available datasets under the 'BeIR' organization.
        """
        try:
            from huggingface_hub import list_datasets
            print("Fetching available BeIR datasets from Hugging Face...")
            beir_datasets = list(list_datasets(author='BeIR'))
            
            print(f"Number of available BeIR datasets: {len(beir_datasets)}")
            print("-" * 60)
            print(f"{'Dataset Name':<40}")
            print("-" * 60)
            
            for dataset in beir_datasets:
                # dataset.id looks like 'BeIR/scifact'
                name = dataset.id.split('/')[-1]
                print(f"{name:<40}")
            print("-" * 60)
            print("Note: Detailed statistics (doc/query counts) are not available via API without downloading.")
            
        except ImportError:
            print("huggingface_hub library not found. Please install it to list datasets: pip install huggingface_hub")
            # Fallback to the hardcoded list if HF hub is not available
            print("\nFallback: Known BeIR datasets (statistics from documentation):")
            datasets = [
                "msmarco", "trec-covid", "nfcorpus", "bioasq", "nq", "hotpotqa", "fiqa", 
                "signal1m", "trec-news", "robust04", "arguana", "webis-touche2020", 
                "cqadupstack", "quora", "dbpedia-entity", "scidocs", "fever", 
                "climate-fever", "scifact"
            ]
            for name in datasets:
                print(f"- {name}")

if __name__ == "__main__":
    # Example usage
    DataHandler.print_available_datasets()
    quora = DataHandler("trec-covid")
    quora.download_and_cache()