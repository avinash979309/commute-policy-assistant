FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

# requirements.lock pins every package (not just the 7 we use directly), so the
# image builds the same way next month as it does today.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

# Download the embedding model at build time, so the first start is fast and
# needs no internet for embeddings.
RUN python -c "from chromadb.utils.embedding_functions import DefaultEmbeddingFunction as E; E()(['warm up'])"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
