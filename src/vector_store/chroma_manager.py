import chromadb
from typing import List, Dict, Any
import hashlib
import logging

logger = logging.getLogger(__name__)

class ChromaManager:
    """Manages connection and operations for the local ChromaDB instance."""
    
    def __init__(self, persist_dir: str, collection_name: str = "multimodal_rag"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        # Using Inner Product (ip) because CLIP embeddings are inherently normalized
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "ip"} 
        )

    def _generate_id(self, chunk: Dict[str, Any]) -> str:
        """Fallback ID generation if the parser didn't provide one."""
        if "id" in chunk:
            return chunk["id"]
        content = str(chunk.get("content", ""))
        return hashlib.md5(content.encode()).hexdigest()

    def add_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]], batch_size: int = 100):
        if not chunks or not embeddings:
            logger.warning("No chunks or embeddings provided to add_chunks.")
            return

        if len(chunks) != len(embeddings):
            logger.error("Mismatch between chunks and embeddings length.")
            return

        total_chunks = len(chunks)

        for i in range(0, total_chunks, batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]

            ids = [self._generate_id(c) for c in batch_chunks]
            documents = [str(c.get("content", "")) for c in batch_chunks]
            metadatas = [c.get("metadata", {}) for c in batch_chunks]

            self.collection.upsert(
                ids=ids,
                embeddings=batch_embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Upserted batch {i//batch_size + 1} ({len(batch_chunks)} chunks).")

    def query(self, query_embedding: List[float], n_results: int = 5) -> Dict[str, Any]:
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )