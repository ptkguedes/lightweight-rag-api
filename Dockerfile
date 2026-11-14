# ChromaDB 0.4.24 pins chroma-hnswlib 0.7.3, which publishes manylinux
# wheels for CPython 3.11 and not for 3.12. 3.12 would compile from source.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# CPU wheels for PyTorch / NumPy need the OpenMP runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

VOLUME ["/app/chroma_db"]

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
