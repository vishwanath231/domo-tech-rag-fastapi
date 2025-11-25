import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Any, Dict
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    
    def _load_model(self):
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def generate_embedding(self, texts: List[str]) -> np.ndarray:
        """
            Generate embedding for a list of text

        args:
            texts: List of text strings to embed

        returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """

        if not self.model:
            raise ValueError("Model not loaded")
        
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings   