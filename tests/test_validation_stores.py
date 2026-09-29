import os

os.environ.setdefault("MISTRAL_API_KEY", "test-key")
os.environ.setdefault("HUGGINGFACE_API_KEY", "test-key")

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

import src.vectorstore.chroma_store as chroma_module
from src.vectorstore.bm25_store import BM25Store
from src.vectorstore.chroma_store import ChromaStore


class TinyEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


def test_chroma_accepts_local_embedding_backend(tmp_path, monkeypatch):
    def fail_if_called():
        raise AssertionError("external embedding service should not be initialized")

    monkeypatch.setattr(chroma_module, "get_embedding_service", fail_if_called)

    store = ChromaStore(
        "injected_embeddings",
        embeddings=TinyEmbeddings(),
        persist_directory=tmp_path / "chroma",
    )
    document = Document(
        page_content="judgment text",
        metadata={
            "source_type": "judgment",
            "chunk_id": "doc:p1:c0",
        },
    )

    ids = store.add_documents([document])

    assert ids == ["doc:p1:c0"]
    assert store.get_collection_stats()["document_count"] == 1


def test_bm25_accepts_isolated_persist_directory(tmp_path):
    store = BM25Store(
        "isolated",
        persist_directory=tmp_path / "bm25",
    )
    document = Document(
        page_content="Civil Revision No 205 of 2021",
        metadata={
            "source_type": "judgment",
            "chunk_id": "doc:p1:c0",
        },
    )

    store.add_documents([document])
    store.save()

    assert store.persist_path.parent == tmp_path / "bm25"
    assert store.persist_path.exists()
