from abc import ABC, abstractmethod
from typing import List, Dict, Union
import numpy as np

class IRExpert(ABC):
    @abstractmethod
    def build_index(self, corpus: Dict[str, Dict[str, str]]) -> None:
        pass
        
    @abstractmethod
    def save_index(self, path: str) -> None:
        pass
        
    @abstractmethod
    def load_index(self, path: str) -> None:
        pass
        
    @abstractmethod
    def search(self, query: Union[str, List[str]], top_k: int = 10) -> List[List[str]]:
        pass

    @abstractmethod
    def score_pairs(self, query: str, doc_ids: List[str]) -> np.ndarray:
        pass
