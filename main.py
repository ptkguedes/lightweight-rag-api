from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import chromadb
from sentence_transformers import SentenceTransformer


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializa o modelo de embedding leve e o cliente do ChromaDB local
    # no startup da aplicação, e não na importação do módulo.
    app.state.model = SentenceTransformer("all-MiniLM-L6-v2")
    app.state.chroma_client = chromadb.PersistentClient(path="./chroma_db")
    app.state.collection = app.state.chroma_client.get_or_create_collection(
        name="knowledge_base"
    )
    yield


app = FastAPI(
    title="Lightweight RAG API",
    version="1.0.0",
    description="A minimalist microservice for Retrieval-Augmented Generation context retrieval.",
    lifespan=lifespan,
)

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
    if len(payload.documents) != len(payload.ids):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Documents and IDs lists must have the same length.",
        )

    try:
        # Gera os embeddings vetoriais dos textos
        embeddings = app.state.model.encode(payload.documents).tolist()

        app.state.collection.add(
            documents=payload.documents,
            embeddings=embeddings,
            ids=payload.ids
        )
        return {"status": "success", "added_count": len(payload.documents)}
    except HTTPException:
        # HTTPException subclasses Exception and would otherwise be rewritten as 500.
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/api/v1/query")
def query_context(payload: QueryInput):
    """Busca os trechos mais relevantes com base na similaridade semântica da query."""
    try:
        query_embedding = app.state.model.encode([payload.query]).tolist()

        results = app.state.collection.query(
            query_embeddings=query_embedding,
            n_results=payload.n_results
        )

        return {
            "query": payload.query,
            "matches": results.get("documents", [])
        }
    except HTTPException:
        # HTTPException subclasses Exception and would otherwise be rewritten as 500.
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.get("/api/v1/health", status_code=status.HTTP_200_OK)
def health_check():
    return {"status": "online", "service": "RAG API running successfully"}
