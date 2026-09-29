import os
from types import SimpleNamespace

os.environ.setdefault("MISTRAL_API_KEY", "test-key")
os.environ.setdefault("HUGGINGFACE_API_KEY", "test-key")

from langchain_core.documents import Document

import src.vectorstore.bm25_store as bm25_module


def test_reingesting_same_chunk_replaces_instead_of_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bm25_module,
        "get_settings",
        lambda: SimpleNamespace(bm25_persist_dir=tmp_path),
    )
    store = bm25_module.BM25Store("cases")

    first = Document(
        page_content="first version",
        metadata={"source_type": "judgment", "chunk_id": "doc:p1:c0"},
    )
    second = Document(
        page_content="updated version",
        metadata={"source_type": "judgment", "chunk_id": "doc:p1:c0"},
    )

    store.add_documents([first])
    store.add_documents([second])

    assert len(store.documents) == 1
    assert store.documents[0].page_content == "updated version"


def test_bm25_accepts_chroma_style_filters(tmp_path, monkeypatch):
    monkeypatch.setattr(
        bm25_module,
        "get_settings",
        lambda: SimpleNamespace(bm25_persist_dir=tmp_path),
    )
    store = bm25_module.BM25Store("cases")
    store.add_documents(
        [
            Document(
                page_content="property appeal",
                metadata={
                    "source_type": "judgment",
                    "court": "High Court Division",
                    "page_start": 4,
                    "chunk_id": "one",
                },
            ),
            Document(
                page_content="property appeal",
                metadata={
                    "source_type": "judgment",
                    "court": "Appellate Division",
                    "page_start": 8,
                    "chunk_id": "two",
                },
            ),
        ]
    )

    results = store.search(
        "property",
        k=5,
        filter={
            "$and": [
                {"court": {"$eq": "High Court Division"}},
                {"page_start": {"$gte": 1}},
            ]
        },
    )

    assert len(results) == 1
    assert results[0].metadata["chunk_id"] == "one"
