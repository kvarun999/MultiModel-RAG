# -------- Base Image --------
FROM python:3.11-slim

# -------- System Dependencies --------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------- Set Work Directory --------
WORKDIR /app

# -------- Copy Requirements First (for caching) --------
COPY requirements.txt .

# -------- Install Python Dependencies --------
RUN pip install --no-cache-dir -r requirements.txt

# -------- Copy Project Files --------
COPY . .

# -------- Expose Port --------
EXPOSE 8000

# -------- Start API --------
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]