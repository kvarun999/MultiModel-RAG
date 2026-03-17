from sentence_transformers import SentenceTransformer
from PIL import Image
from typing import List
import logging

logger = logging.getLogger(__name__)


class MultimodalEmbedder:
    """Loads a local model (like CLIP) to embed both text and images into the same vector space."""

    def __init__(self, model_name: str):
        logger.info(f"Loading embedding model: {model_name} (This may take a moment...)")

        # CPU-only (no CUDA logic needed)
        self.model = SentenceTransformer(model_name)

        # Optional but safe
        self.model.eval()

    def embed_text(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def embed_images(self, image_paths: List[str]) -> List[List[float]]:
        if not image_paths:
            return []

        valid_images = []

        for path in image_paths:
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB")
                    valid_images.append(img.copy())
            except Exception as e:
                logger.warning(f"Failed to load image at {path}: {e}")

        if not valid_images:
            return []

        embeddings = self.model.encode(
            valid_images,
            batch_size=16,
            convert_to_numpy=True
        )

        return embeddings.tolist()