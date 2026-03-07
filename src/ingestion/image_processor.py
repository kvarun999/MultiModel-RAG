# Image processor
# TODO: Implement image processing logic
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
import pytesseract
from src.ingestion.document_parser import BaseParser

class ImageParser(BaseParser):
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        chunks = []
        document_id = file_path.name

        img = Image.open(file_path).convert("RGB")
        img.thumbnail((2000, 2000)) # Optimize for OCR speed

        # -------- 1. EXTRACT TEXT --------
        gray = img.convert("L")
        text = pytesseract.image_to_string(gray)

        if text.strip():
            chunks.append({
                "content": text,
                "metadata": {
                    "document_id": document_id,
                    "page_number": 1, # Hardcoded because a standalone image is effectively one page
                    "content_type": "text"
                }
            })

        # -------- 2. RETAIN IMAGE CONTEXT --------
        chunks.append({
            "content": str(file_path),
            "metadata": {
                "document_id": document_id,
                "page_number": 1,
                "content_type": "image"
            }
        })

        return chunks