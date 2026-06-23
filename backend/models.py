# -*- coding: utf-8 -*-
"""AIKP Data Models — Turn log, session, entity state, world book schemas."""

from __future__ import annotations
from typing import TypedDict, Optional, Union
import time


# ── Turn Log ───────────────────────────────────────────────────

class TurnLog(TypedDict, total=False):
    """Immutable record of one game turn. Never deleted, never LLM-compressed."""
    turn: int
    scene: str
    timestamp: float
    player_input: str
    gm_response: str
    dice_result: Optional[dict]          # None if no check was made
    entity_state_changes: dict           # {entity_id: "old_state→new_state"}
    npc_changes: dict                    # {npc_id: {field: new_value}}
    new_flags: list[str]                 # e.g. ["altar_searched"]
    items_obtained: list[str]
    items_used: list[str]
    scene_transition: Optional[str]      # None if no movement


# ── Entity Memory ──────────────────────────────────────────────

class EntityMemory(TypedDict):
    """One recorded interaction with an entity, keyed by entity_id."""
    entity_id: str
    turn: int
    type: str           # "dialogue" | "state_change" | "discovery" | "decision"
    summary: str
    importance: float   # 0.0–1.0, key events = 1.0
    scene: str


# ── Player State ───────────────────────────────────────────────

class PlayerState(TypedDict, total=False):
    current_scene: str
    inventory: list[str]
    skills: dict[str, int]      # {"侦查": 60, "图书馆使用": 70, ...} (attrs+skills)
    hp: int
    max_hp: int
    san: int
    max_san: int
    mp: int
    max_mp: int
    luck_tokens: int            # 大幸运 token (0 or 1)
    name: str                   # 调查员姓名 (from imported card)
    profession: str             # 职业
    attributes: dict[str, int]  # 九大属性快照 {力量, 敏捷, ...}
    background: str             # 背景故事 (injected into KP context)


# ── NPC Disposition ────────────────────────────────────────────

class NPCDisposition(TypedDict):
    npc_id: str
    value: int            # -100 .. +100
    last_change_turn: int


# ── Session ────────────────────────────────────────────────────

class Session(TypedDict, total=False):
    chat_id: str
    model: str                     # world book name
    created_at: float
    updated_at: float
    current_turn: int
    player_state: PlayerState
    entity_states: dict[str, str]  # {entity_id: current_state}
    entity_states_cooldown: dict[str, int]  # {entity_id: cooldown_until_turn}
    flags: list[str]
    npc_dispositions: dict[str, int]  # {npc_id: value}
    npc_states: dict[str, dict]      # {npc_merged_name: {static, dynamic}}  (see npc_state.py)
    entity_memories: dict[str, list[EntityMemory]]  # {entity_id: [memory, ...]}
    turn_log: list[TurnLog]
    plot_phase: str               # "intro" | "investigation" | "climax" | "resolution"
    derailment_level: int         # 0–4 escalation counter
    rule_system: str              # "dnd" | "coc", from world book
    _story_summary: str           # LLM-compressed prior scene narrative
    _cached_summary: str          # mid-range conversation summary cache
    _cached_summary_turn: int     # turn when _cached_summary was last generated
    _pending_san_result: Optional[dict]  # auto-SAN result from last turn
    imported_card: Optional[dict]        # parsed .st character card (survives reset)
    pending_check: Optional[dict]        # check awaiting the player's dice roll (/api/roll)
    discovered_clues: list[str]          # scene clue IDs discovered so far
    current_beat_id: str                 # active story beat ID
    completed_beats: list[str]           # beat IDs already completed
    unlocked_scenes: list[str]           # scenes explicitly unlocked by beat progression
    companions: list[str]                # NPC entity ids currently travelling with the player


# ── Entity State Definition (world book level) ─────────────────

class EntityStateDef(TypedDict, total=False):
    """Definition of one state within an entity's state machine."""
    description: str
    triggers: list[str]                    # keywords that activate this state
    check: Optional[dict]                  # {"skill": "Perception", "dc": 14}
    san_check: Optional[str]               # CoC SAN loss e.g. "1/1d3", "1d3/1d6"
    on_pass: Optional[dict]               # {"to_state": "...", "narration": "..."}
    on_fail: Optional[dict]               # {"to_state": "...", "narration": "...", "cooldown_turns": int}
    on_trigger: Optional[dict]            # {"to_state": "...", "narration": "..."}  (no check)
    repeatable: bool                       # True = trigger doesn't change state
    dialogue: Optional[dict[str, str]]     # NPC: {"topic": "response"}


class EntityDef(TypedDict):
    """Full entity definition in world book."""
    id: str
    type: str              # "clue" | "item" | "npc" | "door" | "container"
    name: str
    scene: str
    initial_state: str
    states: dict[str, EntityStateDef]
    description: str       # general description (fallback)
    personality: Optional[str]   # NPC only
    significance: Optional[str]  # "key" | "clue" | "decoration"


# ── NPC State (see docs/npc-design.md) ────────────────────────

class NpcStyle(TypedDict):
    verbosity: str          # "many_words" | "normal" | "few_words" | "grunt"
    tone: str               # "cheerful" | "nervous" | "gruff" | "academic" | "neutral"
    initiative: str         # "active" | "passive"


class NpcDialogueTopic(TypedDict):
    text: str               # what NPC reveals when this topic is discussed
    trust_required: int     # minimum trust to unlock (signal, not hard gate)
    requires_flag: Optional[str]  # prerequisite flag


class NpcStatic(TypedDict):
    name: str
    profession: str
    appearance: str
    personality: str
    style: NpcStyle
    dialogue: dict[str, NpcDialogueTopic]


class NpcRevealedEntry(TypedDict):
    topic: str
    turn: int
    summary: str            # what was actually told to the player


class NpcInteraction(TypedDict):
    turn: int
    player: str             # brief player action
    response: str           # brief NPC response
    delta: int              # trust change
    revealed: Optional[str]  # topic revealed (if any)


class NpcDynamic(TypedDict):
    stage: str              # "hostile"|"stranger"|"cautious"|"trusting"|"close"
    trust: int              # -100 ~ +100
    mood: str               # current emotional state
    revealed: list[NpcRevealedEntry]
    interactions: list[NpcInteraction]
    summary: str            # LLM-compressed old interactions


# ── Story Beat ─────────────────────────────────────────────────

class StoryBeat(TypedDict, total=False):
    """One narrative milestone in the module's investigation chain."""
    id: str
    name: str                       # "第一幕：初到公馆"
    kp_note: str                    # KP-only: what this beat is about, what players should find
    scenes: list[str]               # scene_ids available/relevant in this beat
    critical_clues: list[str]       # clue IDs — finding these advances the beat
    optional_clues: list[str]       # good to find but not required
    advance_when: str               # "any_critical" | "all_critical" | "visited"
    unlocks_scenes: list[str]       # scene_ids newly available after this beat completes
    unlocks_npcs: list[str]         # NPC ids that open up after beat


# ── World Book ─────────────────────────────────────────────────

class SceneDef(TypedDict):
    id: str
    name: str
    description: str
    atmosphere: Optional[str]
    exits: Union[dict[str, str], list[str]]  # {"keyword": "scene_id"} or ["scene_id", ...]
    goals: Optional[list[str]]     # Scene objectives for GM


class WorldBook(TypedDict):
    name: str
    version: str
    description: str
    rule_system: Optional[str]      # "dnd" | "coc"
    scenes: dict[str, SceneDef]
    entities: dict[str, EntityDef]
    plot_outline: Optional[list[str]]


# ── Pre-built Indices ──────────────────────────────────────────

class SceneIndex(TypedDict):
    """scene_id → list of entity_ids in that scene."""
    pass  # dict[str, list[str]]


class EntityIndex(TypedDict):
    """entity_id → {type, scene, name} for O(1) lookup."""
    pass  # dict[str, {"type": str, "scene": str, "name": str}]


# ── Context Assembly ───────────────────────────────────────────

class AssembledContext(TypedDict):
    """Result of context assembly, ready for LLM prompt."""
    system_prompt: str      # Layer 0: GM rules, always cached
    state_snapshot: str     # Layer 1: computed state summary
    entity_memories: str    # Layer 2: relevant memories for current scene
    conversation: str       # Layer 3: recent turns + summary
    player_turn: str        # Layer 4: current user message


# ── Session Factory ────────────────────────────────────────────

def create_session(chat_id: str, model: str) -> Session:
    now = time.time()
    return Session(
        chat_id=chat_id,
        model=model,
        created_at=now,
        updated_at=now,
        current_turn=0,
        player_state=PlayerState(
            current_scene="",
            inventory=[],
            skills={"Perception": 5, "Strength": 3, "Persuasion": 2},
            hp=10, max_hp=10,
            san=60, max_san=60,
            mp=0, max_mp=0,
            luck_tokens=0,
            name="", profession="", attributes={}, background="",
        ),
        entity_states={},
        entity_states_cooldown={},
        flags=[],
        npc_dispositions={},
        npc_states={},
        entity_memories={},
        turn_log=[],
        plot_phase="intro",
        derailment_level=0,
        rule_system="dnd",
        _story_summary="",
        _cached_summary="",
        _cached_summary_turn=0,
        _pending_san_result=None,
        imported_card=None,
        pending_check=None,
        discovered_clues=[],
        current_beat_id="",
        completed_beats=[],
        unlocked_scenes=[],
        companions=[],
    )
