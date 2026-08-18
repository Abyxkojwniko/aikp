# -*- coding: utf-8 -*-
"""AIKP RAG Engine — Per-module ChromaDB collections for semantic retrieval."""

from __future__ import annotations

import os
from typing import Optional

import chromadb
from chromadb.config import Settings

from config import WORLD_BOOK_DIR


# ── ChromaDB Setup ──────────────────────────────────────────

_client: Optional[chromadb.ClientAPI] = None


def _get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        chroma_dir = os.path.join(WORLD_BOOK_DIR, "_chroma")
        os.makedirs(chroma_dir, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _collection_name(module_name: str) -> str:
    """Sanitize module name into a valid ChromaDB collection name."""
    safe = module_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return f"aikp_{safe}"[:63]  # ChromaDB limit


# ── Public API ──────────────────────────────────────────────

def delete_collection(module_name: str) -> None:
    try:
        _get_client().delete_collection(_collection_name(module_name))
    except Exception:
        pass


def delete_all() -> None:
    for name in list_collections():
        try:
            _get_client().delete_collection(name)
        except Exception:
            pass


def list_collections() -> list[str]:
    return _get_client().list_collections()


def index_world_book(module_name: str, world: dict) -> int:
    """Index all entity descriptions from a world book into its own collection.
    Returns the number of documents indexed."""
    collection = _get_client().get_or_create_collection(
        name=_collection_name(module_name),
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []

    entities = world.get("entities", {})
    if not isinstance(entities, dict):
        return 0

    for eid, entity in entities.items():
        if not isinstance(entity, dict):
            continue

        etype = entity.get("type", "unknown")
        ename = entity.get("name", eid)
        desc = entity.get("description", "") or entity.get("desc", "")
        scene = entity.get("scene", "")

        # Build a rich text representation for embedding
        text_parts = [f"[{module_name}] {etype}: {ename}"]
        if desc:
            text_parts.append(desc)
        if entity.get("appearance"):
            text_parts.append(f"Appearance: {entity['appearance']}")
        if entity.get("personality"):
            text_parts.append(f"Personality: {entity['personality']}")
        if entity.get("source_quote"):
            text_parts.append(f"Canonical source: {entity['source_quote']}")
        states = entity.get("states", {})
        if isinstance(states, dict):
            for sname, sdef in states.items():
                if isinstance(sdef, dict):
                    sd_desc = sdef.get("description", "")
                    if sd_desc:
                        text_parts.append(f"State {sname}: {sd_desc}")

        document = "\n".join(text_parts)
        ids.append(eid)
        documents.append(document)
        metadatas.append({
            "type": etype,
            "name": ename,
            "scene": scene,
        })

    for sid, scene in world.get("scenes", {}).items():
        if not isinstance(scene, dict):
            continue
        source = scene.get("source_text", "")
        desc = scene.get("desc", "") or scene.get("description", "")
        if not source and not desc:
            continue
        name = scene.get("name", sid)
        ids.append(f"scene:{sid}")
        documents.append(
            f"[{module_name}] scene: {name}\n"
            f"{source or desc}"
        )
        metadatas.append({
            "type": "scene",
            "name": name,
            "scene": sid,
        })

    if not ids:
        return 0

    # Upsert handles both insert and update
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    return len(ids)


def query(
    module_name: str,
    query_text: str,
    n_results: int = 5,
    filter_type: Optional[str] = None,
    filter_scene: Optional[str] = None,
) -> list[dict]:
    """Semantic search within a module's collection.
    Returns list of {id, document, metadata, distance}."""
    try:
        collection = _get_client().get_collection(_collection_name(module_name))
    except Exception:
        return []

    # Build optional filter
    where = {}
    if filter_type:
        where["type"] = filter_type
    if filter_scene:
        where["scene"] = filter_scene

    results = collection.query(
        query_texts=[query_text],
        n_results=n_results,
        where=where if where else None,
        include=["documents", "metadatas", "distances"],
    )

    # Flatten ChromaDB response
    items = []
    if results.get("ids") and results["ids"][0]:
        for i in range(len(results["ids"][0])):
            items.append({
                "id": results["ids"][0][i],
                "document": results["documents"][0][i] if results.get("documents") else "",
                "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                "distance": results["distances"][0][i] if results.get("distances") else 0.0,
            })
    return items


def hybrid_search(
    module_name: str,
    query_text: str,
    current_scene: str,
    n_results: int = 5,
) -> list[dict]:
    """Retrieve canonical facts from the current scene only.

    Searching the entire module can surface a future clue or antagonist secret
    before the player reaches that scene. Cross-scene continuity is supplied by
    the deterministic session log and matched-NPC memory instead.
    """
    return query(
        module_name,
        query_text,
        n_results=n_results,
        filter_scene=current_scene,
    )


def summary(module_name: str) -> dict:
    try:
        collection = _get_client().get_collection(_collection_name(module_name))
        return {"collection": _collection_name(module_name), "count": collection.count()}
    except Exception:
        return {"collection": _collection_name(module_name), "count": 0}
