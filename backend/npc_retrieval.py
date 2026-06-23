# -*- coding: utf-8 -*-
"""AIKP NPC Retrieval — multi-stage search for NPC interaction history.

Design: docs/npc-design.md

Retrieval cascade:
  1. search_recent()  — keyword match in last 5 turn_log entries
  2. search_history() — keyword match across full turn_log
  3. search_rag()     — semantic search in entity_memories via ChromaDB
  4. Empty            — no match found
"""

from __future__ import annotations


def search_recent(keyword: str, session: dict, n: int = 5) -> str:
    """Search last N turn_log entries for keyword match.

    Returns matched turns as formatted text, or empty string.
    """
    log = session.get("turn_log", [])
    if not log:
        return ""

    kw = keyword.lower()
    recent = log[-n:]
    results = []

    for entry in recent:
        t = entry.get("turn", "?")
        player_input = entry.get("player_input", "")
        gm_response = entry.get("gm_response", "")

        if kw in player_input.lower() or kw in gm_response.lower():
            lines = [f"[T{t}] Player: {player_input[:200]}"]
            if gm_response:
                lines.append(f"      GM: {gm_response[:300]}")
            results.append("\n".join(lines))

    if not results:
        return ""

    return "=== RETRIEVED (recent) ===\n" + "\n\n".join(results[-3:])  # max 3


def search_history(keyword: str, session: dict) -> str:
    """Search full turn_log history for keyword match.

    Skips entries already covered by search_recent.
    Returns matched turns as formatted text, or empty string.
    """
    log = session.get("turn_log", [])
    if len(log) <= 5:
        return ""  # already covered by search_recent

    kw = keyword.lower()
    # Search older entries only
    older = log[:-5]
    results = []

    for entry in older:
        t = entry.get("turn", "?")
        player_input = entry.get("player_input", "")
        gm_response = entry.get("gm_response", "")

        if kw in player_input.lower() or kw in gm_response.lower():
            lines = [f"[T{t}] Player: {player_input[:200]}"]
            if gm_response:
                lines.append(f"      GM: {gm_response[:300]}")
            results.append("\n".join(lines))

    if not results:
        return ""

    return "=== RETRIEVED (history) ===\n" + "\n\n".join(results[-3:])


def search_rag(keyword: str, session: dict,
               entity_index: dict[str, dict]) -> str:
    """RAG semantic search in entity_memories via ChromaDB.

    Falls back to simple keyword search if RAG unavailable.
    Returns matched memory entries, or empty string.
    """
    memories = session.get("entity_memories", {})
    if not memories:
        return ""

    kw = keyword.lower()
    results = []

    # Simple keyword search across all entity memories
    for eid, mems in memories.items():
        einfo = entity_index.get(eid, {})
        ename = einfo.get("name", eid)
        for mem in mems:
            summary = mem.get("summary", "")
            if kw in summary.lower() or kw in ename.lower():
                results.append(
                    f"  [{ename}] T{mem.get('turn', '?')}: {summary[:150]}"
                )

    if not results:
        return ""

    return "=== RETRIEVED (memory) ===\n" + "\n".join(results[-5:])


def retrieve(keyword: str, session: dict,
             entity_index: dict[str, dict]) -> str:
    """Retrieve relevant interaction history for a keyword.

    Cascade: recent -> history -> RAG -> empty.
    """
    if not keyword or not keyword.strip():
        return ""

    kw = keyword.strip()

    # Stage 1: recent
    result = search_recent(kw, session, n=5)
    if result:
        return result

    # Stage 2: history
    result = search_history(kw, session)
    if result:
        return result

    # Stage 3: RAG
    result = search_rag(kw, session, entity_index)
    if result:
        return result

    # Stage 4: no match
    return ""
