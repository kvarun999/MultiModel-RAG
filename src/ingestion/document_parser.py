from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
import os
import pandas as pd

# The Trifecta
import fitz
from PIL import Image
import pytesseract
import io


class BaseParser(ABC):
    """
    Abstract interface for all document parsers.
    Enforces a strict return schema for vector DB ingestion.
    """
    @abstractmethod
    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        pass


class PDFParser(BaseParser):
    def __init__(self, image_output_dir: Path):
        self.image_output_dir = image_output_dir
        # Ensure the output directory exists before processing starts
        os.makedirs(self.image_output_dir, exist_ok=True)

    def parse(self, file_path: Path) -> List[Dict[str, Any]]:
        chunks = []
        # Track image references to prevent extracting the same header logo on every page
        seen_xrefs = set()

        doc = fitz.open(file_path)

        for page in doc:
            # Improvement: PyMuPDF uses 0-based indexing for pages.
            # Adding +1 ensures page numbers match what users expect in real documents.
            page_number = page.number + 1

            document_id = file_path.name

            # -------- 1. TEXT EXTRACTION --------
            text = page.get_text()

            if text.strip():
                # Born-digital PDF text found
                chunks.append({
                    "content": text,
                    "metadata": {
                        "document_id": document_id,
                        "page_number": page_number,
                        "content_type": "text"
                    }
                })
            else:
                # Fallback: OCR the entire page if it's a scanned document
                pix = page.get_pixmap()
                page_img = Image.open(io.BytesIO(pix.tobytes())).convert("RGB")
                ocr_text = pytesseract.image_to_string(page_img)

                if ocr_text.strip():
                    chunks.append({
                        "content": ocr_text,
                        "metadata": {
                            "document_id": document_id,
                            "page_number": page_number,
                            "content_type": "text"
                        }
                    })

            # -------- 2. IMAGE EXTRACTION --------
            img_list = page.get_images(full=True)

            # IMAGE EXTRACTION FIX
            for img in img_list:
                xref = img[0]
                if xref in seen_xrefs:
                    continue
                seen_xrefs.add(xref)

                try:
                    image = doc.extract_image(xref)
                    image_bytes = image["image"]

                    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                except Exception:
                    continue

                pil_img.thumbnail((2000, 2000))

                image_path = self.image_output_dir / f"{file_path.stem}_p{page_number}_img{xref}.png"
                pil_img.save(image_path)

                # Store the *path* to the image for the Vector DB and VLM
                chunks.append({
                    "content": str(image_path),
                    "metadata": {
                        "document_id": document_id,
                        "page_number": page_number,
                        "content_type": "image"
                    }
                })

                # OCR any text trapped inside the extracted image (e.g., text in a diagram)
                gray = pil_img.convert("L")
                ocr_text = pytesseract.image_to_string(gray)

                if ocr_text.strip():
                    chunks.append({
                        "content": ocr_text,
                        "metadata": {
                            "document_id": document_id,
                            "page_number": page_number,
                            "content_type": "text"
                        }
                    })

            # -------- 3. TABLE EXTRACTION --------
            try:
                tables = page.find_tables()
            except Exception:
                tables = None

            for table in tables.tables:
                data = table.extract()
                if not data:
                    continue

                # Clean the table: remove entirely empty rows
                data = [row for row in data if any(cell for cell in row)]
                if len(data) < 2:
                    continue # Skip invalid tables (e.g., just a single header row)

                # Convert to Markdown so the VLM can understand the structure natively
                df = pd.DataFrame(data[1:], columns=data[0])
                table_markdown = df.to_markdown(index=False)

                chunks.append({
                    "content": table_markdown,
                    "metadata": {
                        "document_id": document_id,
                        "page_number": page_number,
                        "content_type": "table"
                    }
                })

        doc.close()
        return chunks
