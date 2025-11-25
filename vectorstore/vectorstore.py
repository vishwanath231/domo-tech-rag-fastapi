import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Any, Dict
from sklearn.metrics.pairwise import cosine_similarity
import os

class VectorStore:
    def __init__(self, collection_name: str = "md_documents", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            # create chromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"Description": "md Document Embeddings for RAG"}
            )

        except Exception as e:
            print(f"Error initializing store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")
        
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []

        # prepare data for insertion
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            # prepare document text
            documents_text.append(doc.page_content)

            # Embedding
            embeddings_list.append(embedding.tolist())

        try:
            # add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
        except Exception as e:
            print(f"Error preparing documents for insertion: {e}")
            raise