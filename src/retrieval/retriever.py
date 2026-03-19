from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import logging

logger = logging.getLogger(__name__)

class MultimodalRetriever:
    """Query-time retrieval combining CLIP embeddings and CrossEncoder reranking."""

    def __init__(self, embedder, vector_store, reranker_model_name: str):
        self.embedder = embedder
        self.vector_store = vector_store
        self.reranker = CrossEncoder(reranker_model_name)

    def retrieve(self, query: str, top_k: int = 5, distance_threshold: float = 0.5) -> List[Dict[str, Any]]:
        if not query:
            return []

        logger.info(f"Retrieving context for query: '{query}'")

        # 1. Embed query and fetch initial candidates
        embeddings = self.embedder.embed_text([query])
        if not embeddings:
            logger.warning("Query embedding failed.")
            return []

        raw_results = self.vector_store.query(
            query_embedding=embeddings[0],
            n_results=top_k * 4
        )

        formatted_results = []
        if raw_results and raw_results.get("ids") and raw_results["ids"][0]:
            ids = raw_results["ids"][0]
            docs = raw_results["documents"][0]
            metas = raw_results["metadatas"][0]
            
            raw_distances = raw_results.get("distances")
            distances = raw_distances[0] if raw_distances else []

            for i in range(len(ids)):
                dist = distances[i] if i < len(distances) else None
                if dist is not None and dist > distance_threshold:
                    continue

                formatted_results.append({
                    "document_id": metas[i].get("document_id"),
                    "page_number": metas[i].get("page_number"),
                    "content_type": metas[i].get("content_type"),
                    "snippet": docs[i],
                    "distance": dist
                })

        # 2. Modality-Aware Re-ranking
        if formatted_results:
            text_chunks = [r for r in formatted_results if r["content_type"] in ["text", "table"]]
            image_chunks = [r for r in formatted_results if r["content_type"] == "image"]

            # Rerank text/tables via CrossEncoder
            if text_chunks:
                pairs = [(query, r["snippet"]) for r in text_chunks]
                scores = self.reranker.predict(pairs)
                for i, score in enumerate(scores):
                    text_chunks[i]["rerank_score"] = float(score)

            # Interleave images using artificial scoring (trusting CLIP)
            for i, img in enumerate(image_chunks):
                img["rerank_score"] = 5.0 - i 

            # Combine and sort by new scores
            formatted_results = text_chunks + image_chunks
            formatted_results.sort(key=lambda x: x["rerank_score"], reverse=True)

        # 3. Deduplicate based on source and modality
        unique_results = []
        seen_sources = set()

        for r in formatted_results:
            key = (r["document_id"], r["page_number"], r["content_type"])
            if key not in seen_sources:
                unique_results.append(r)
                seen_sources.add(key)
            if len(unique_results) >= top_k:
                break

        logger.info(f"Retrieved {len(unique_results)} relevant chunks after reranking")
        return unique_results