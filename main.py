from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer

app = FastAPI(
    title="Lightweight RAG API",
    version="1.0.0",
    description="A minimalist microservice for Retrieval-Augmented Generation context retrieval."
)

# Inicializa o modelo de embedding leve e o cliente do ChromaDB local
model = SentenceTransformer("all-MiniLM-L6-v2")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="knowledge_base")

# Schemas de Entrada (Pydantic)
class DocumentInput(BaseModel):
    documents: list[str]
    ids: list[str]

class QueryInput(BaseModel):
    query: str
    n_results: int = 2

@app.post("/api/v1/documents", status_code=status.HTTP_201_CREATED)
def add_documents(payload: DocumentInput):
    """Ingere novos documentos na base vetorial local."""
    try:
        if len(payload.documents) != len(payload.ids):
            raise HTTPException(status_code=400, detail="Documents and IDs lists must have the same length.")
        
        # Gera os embeddings vetoriais dos textos
        embeddings = model.encode(payload.documents).tolist()
        
        collection.add(
            documents=payload.documents,
            embeddings=embeddings,
            ids=payload.ids
        )
        return {"status": "success", "added_count": len(payload.documents)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/query")
def query_context(payload: QueryInput):
    """Busca os trechos mais relevantes com base na similaridade semântica da query."""
    try:
        query_embedding = model.encode([payload.query]).tolist()
        
        results = collection.query(
            query_embeddings=query_embedding,
            n_results=payload.n_results
        )
        
        return {
            "query": payload.query,
            "matches": results.get("documents", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "online", "service": "RAG API running successfully"}
