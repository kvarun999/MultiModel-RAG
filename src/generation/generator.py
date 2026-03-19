import base64
from typing import List, Dict, Any
import logging
from huggingface_hub import AsyncInferenceClient
import os

logger = logging.getLogger(__name__)

class MultimodalGenerator:
    def __init__(self, provider: str, api_key: str):
        """Initializes the VLM client. No hardcoded tokens allowed."""
        self.provider = provider

        # Using Async client for high-performance FastAPI integration
        self.client = AsyncInferenceClient(
            model="llava-hf/llava-1.5-7b-hf",
            token=api_key
        )

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Reads a local file and converts it to a base64 string."""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            return ""

    def _detect_mime(self, image_path: str) -> str:
        """Detect MIME type based on file extension."""
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            return "image/png"
        if ext in [".jpg", ".jpeg"]:
            return "image/jpeg"
        return "image/png"

    async def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:

        # 1. Prepare the content blocks for the VLM
        content_blocks = []
        text_context_str = "PROVIDED CONTEXT:\n"
        image_count = 0
        max_images = 3  # Prevent sending too many images to API

        for chunk in context_chunks:
            if chunk["content_type"] in ["text", "table"]:
                text_context_str += (
                    f"[Doc: {chunk['document_id']}, Page: {chunk['page_number']}]\n"
                    f"{chunk['snippet']}\n\n"
                )

            elif chunk["content_type"] == "image" and image_count < max_images:
                base64_img = self._encode_image_to_base64(chunk["snippet"])

                if base64_img:
                    mime = self._detect_mime(chunk["snippet"])
                    content_blocks.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{base64_img}"
                        }
                    })
                    image_count += 1

        # Append compiled text context + query
        content_blocks.append({
            "type": "text",
            "text": f"{text_context_str}\n\nUSER QUESTION: {query}"
        })

        # 2. System Prompt
        system_prompt = (
            "You are an expert document analyst. "
            "Answer the user's question using ONLY the provided text and images. "
            "If the answer is not in the context, say "
            "'I cannot answer this based on the provided documents.' "
            "CRITICAL: You MUST cite sources like "
            "'According to [Doc: report.pdf, Page: 4]' "
            "or 'As shown in the image from page 2'."
        )

        # 3. Construct messages payload
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_blocks}
        ]

        logger.info("Sending multimodal payload to LLaVA via Hugging Face...")

        # 4. Call API
        try:
            response = await self.client.chat_completion(
                messages=messages,
                max_tokens=500,
                temperature=0.1
            )
            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"VLM Generation failed: {e}")
            return "Error: Could not generate response from the Vision-Language Model."