from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import logging

from src.config import get_settings
from src.retrieval.retriever import MultimodalRetriever
from src.embeddings.model_loader import MultimodalEmbedder
from src.vector_store.chroma_manager import ChromaManager
from src.generation.generator import MultimodalGenerator

# Logging setup
logger = logging.getLogger(__name__)

# ==========================================
# Global Initialization (Load once)
# ==========================================
settings = get_settings()

embedder = MultimodalEmbedder(settings.clip_model_name)
chroma_manager = ChromaManager(settings.chroma_persist_dir)
retriever = MultimodalRetriever(
    embedder,
    chroma_manager,
    settings.reranker_model_name
)
generator = MultimodalGenerator(
    settings.vlm_provider,
    settings.vlm_api_key
)

app = FastAPI(title="Multimodal RAG API", version="1.0.0")


# --- Schemas ---
class SourceSnippet(BaseModel):
    document_id: str
    page_number: int
    content_type: str
    snippet: str


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceSnippet]


# --- Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/query", response_model=QueryResponse)
async def query_system(request: QueryRequest):

    # Input validation
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        logger.info(f"Received query: {request.query}")

        # 1. Retrieval
        results = retriever.retrieve(request.query)

        if not results:
            return QueryResponse(
                answer="I could not find any relevant information in the documents.",
                sources=[]
            )

        # 2. Generation
        vlm_result = await generator.generate_answer(request.query, results)

        # 3. Format response
        sources = [
            SourceSnippet(
                document_id=res["document_id"],
                page_number=res["page_number"],
                content_type=res["content_type"],
                snippet=res["snippet"]
            )
            for res in results
        ]

        return QueryResponse(answer=vlm_result, sources=sources)

    except Exception as e:
        logger.error(f"API error: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")