from typing import List, Any, Dict
from embedding.embedding import EmbeddingManager
from vectorstore.vectorstore import VectorStore

class RAGRetriever:
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        print(f"Retrieving documents for query: {query}")
        print(f"Top k: {top_k}, Score threshold: {score_threshold}")

        #generate query embedding
        query_embedding = self.embedding_manager.generate_embedding([query])[0]

        # search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )

            #process results
            retrieved_docs = []

            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # convert distance to similarity score (chromaDB uses cosine distance)
                    similarity_score = 1 - distance

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                print(f"Reterieved {len(retrieved_docs)} documents after applying score threshold.")
            else:
                print("No documents retrieved from vector store.")
            
            return retrieved_docs

        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []
