import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from fastapi.testclient import TestClient

import main


class _Vectors:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return self._rows


class StubEmbeddingModel:
    """Deterministic stand-in so tests never download all-MiniLM-L6-v2."""

    def encode(self, texts):
        rows = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            rows.append([byte / 255.0 for byte in digest[:8]])
        return _Vectors(rows)


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        main,
        "SentenceTransformer",
        lambda *args, **kwargs: StubEmbeddingModel(),
    )

    real_client = chromadb.PersistentClient

    def persistent_client(path="./chroma_db", *args, **kwargs):
        return real_client(path=str(tmp_path / "chroma"), *args, **kwargs)

    monkeypatch.setattr(main.chromadb, "PersistentClient", persistent_client)

    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "online",
        "service": "RAG API running successfully",
    }


def test_add_documents(client):
    response = client.post(
        "/api/v1/documents",
        json={
            "documents": [
                "Paris is the capital of France.",
                "The Nile is a river in Africa.",
            ],
            "ids": ["doc-1", "doc-2"],
        },
    )

    assert response.status_code == 201
    assert response.json() == {"status": "success", "added_count": 2}


@pytest.mark.parametrize(
    ("documents", "ids"),
    [
        (["only one"], ["a", "b"]),
        (["a", "b"], ["only one"]),
    ],
)
def test_add_documents_length_mismatch_returns_400(client, documents, ids):
    response = client.post(
        "/api/v1/documents",
        json={"documents": documents, "ids": ids},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Documents and IDs lists must have the same length."
    )


def test_query_returns_stored_document(client):
    added = client.post(
        "/api/v1/documents",
        json={
            "documents": ["Paris is the capital of France."],
            "ids": ["doc-1"],
        },
    )
    assert added.status_code == 201

    response = client.post(
        "/api/v1/query",
        json={"query": "What is the capital of France?", "n_results": 1},
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "What is the capital of France?",
        "matches": [["Paris is the capital of France."]],
    }


def test_import_does_not_initialize_model_or_chroma():
    repo = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
from types import ModuleType

class BoomModel:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("SentenceTransformer was constructed at import")

sentence_transformers = ModuleType("sentence_transformers")
sentence_transformers.SentenceTransformer = BoomModel
sys.modules["sentence_transformers"] = sentence_transformers

class BoomClient:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Chroma client was constructed at import")

chromadb = ModuleType("chromadb")
chromadb.PersistentClient = BoomClient
sys.modules["chromadb"] = chromadb

import main
assert main.app.title == "Lightweight RAG API"
""",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
