"""Observer-specific projections of one canonical physical scene."""

from __future__ import annotations

from copy import deepcopy


def active_observer_id(session: dict) -> str:
    player = session.get("player_state", {})
    value = str(
        player.get("active_character_id")
        or session.get("active_character_id")
        or "player"
    ).strip()
    return value or "player"


def observer_player_state(
    session: dict, observer_id: str = "", *, mutable: bool = False,
) -> dict:
    """Return the active observer's sheet, falling back to the solo player sheet."""
    observer_id = observer_id or active_observer_id(session)
    base = session.setdefault("player_state", {})
    specific = session.get("observer_player_states", {}).get(observer_id)
    if not isinstance(specific, dict):
        return base
    if mutable:
        for key, value in base.items():
            specific.setdefault(key, deepcopy(value))
        return specific
    merged = deepcopy(base)
    for key, value in specific.items():
        if key in {"attributes", "skills"} and isinstance(value, dict):
            merged.setdefault(key, {}).update(value)
        else:
            merged[key] = deepcopy(value)
    return merged


def ensure_perception_state(session: dict) -> None:
    session.setdefault("active_perception_layers", {})
    session.setdefault("perception_events", [])
    session.setdefault("perception_event_seq", 0)


def _world_layers(world: dict) -> list[dict]:
    return [
        row for row in world.get("perception_layers", []) or []
        if isinstance(row, dict) and row.get("id")
    ]


def _layer_by_id(world: dict, layer_id: str) -> dict:
    return next((
        row for row in _world_layers(world)
        if str(row.get("id", "")) == layer_id
    ), {})


def activate_perception_layer(
    session: dict,
    world: dict,
    layer_id: str,
    observer_id: str = "",
    *,
    source: str = "rules",
) -> bool:
    """Activate one authored view for one observer, recording an immutable event."""
    ensure_perception_state(session)
    layer_id = str(layer_id).strip()
    layer = _layer_by_id(world, layer_id)
    if not layer:
        raise ValueError(f"Unknown perception layer: {layer_id}")
    observer_id = observer_id or active_observer_id(session)
    active = session["active_perception_layers"].setdefault(observer_id, {})
    if layer_id in active:
        return False
    active[layer_id] = {
        "layer_id": layer_id,
        "scene_id": str(layer.get("scene_id", "")),
        "activated_turn": int(session.get("current_turn", 0)),
        "source": source,
    }
    session["perception_event_seq"] = int(
        session.get("perception_event_seq", 0)) + 1
    session["perception_events"].append({
        **active[layer_id],
        "seq": session["perception_event_seq"],
        "observer_id": observer_id,
        "type": "perception_layer_activated",
    })
    return True


def deactivate_perception_layer(
    session: dict,
    world: dict,
    layer_id: str,
    observer_id: str = "",
    *,
    source: str = "rules",
) -> bool:
    ensure_perception_state(session)
    layer_id = str(layer_id).strip()
    if not _layer_by_id(world, layer_id):
        raise ValueError(f"Unknown perception layer: {layer_id}")
    observer_id = observer_id or active_observer_id(session)
    active = session["active_perception_layers"].setdefault(observer_id, {})
    if layer_id not in active:
        return False
    active.pop(layer_id)
    session["perception_event_seq"] = int(
        session.get("perception_event_seq", 0)) + 1
    session["perception_events"].append({
        "seq": session["perception_event_seq"],
        "observer_id": observer_id,
        "layer_id": layer_id,
        "turn": int(session.get("current_turn", 0)),
        "source": source,
        "type": "perception_layer_deactivated",
    })
    return True


def active_perception_layers(
    session: dict,
    world: dict,
    scene_id: str = "",
    observer_id: str = "",
) -> list[dict]:
    """Resolve event-activated and stat/flag-conditioned layers for an observer."""
    ensure_perception_state(session)
    observer_id = observer_id or active_observer_id(session)
    scene_id = scene_id or str(
        session.get("player_state", {}).get("current_scene", ""))
    activated = session["active_perception_layers"].get(observer_id, {})
    selected_scenario = str(session.get("current_scenario_id", ""))
    result = []
    for order, layer in enumerate(_world_layers(world)):
        if str(layer.get("scene_id", "")) != scene_id:
            continue
        if (selected_scenario and layer.get("scenario_id")
                and str(layer.get("scenario_id")) != selected_scenario):
            continue
        layer_id = str(layer["id"])
        enabled = layer_id in activated
        activation = str(layer.get("activation", "condition")).lower()
        if not enabled and activation not in {
                "event", "conditional_outcome", "manual"}:
            from conditional_events import condition_satisfied
            enabled = condition_satisfied(
                layer.get("when", {"type": "always"}), session, observer_id)
        if enabled:
            row = deepcopy(layer)
            row["_order"] = order
            result.append(row)
    return sorted(result, key=lambda row: (
        int(row.get("priority", 0)), int(row.get("_order", 0)), str(row["id"])))


def scene_projection(
    session: dict,
    world: dict,
    scene_id: str = "",
    observer_id: str = "",
) -> dict:
    """Return the scene as perceived, without mutating the canonical scene."""
    observer_id = observer_id or active_observer_id(session)
    scene_id = scene_id or str(
        session.get("player_state", {}).get("current_scene", ""))
    scene = world.get("scenes", {}).get(scene_id, {})
    description = str(scene.get("desc") or scene.get("description") or "")
    source_text = str(scene.get("source_text") or scene.get("source_quote") or "")
    projection = {
        "physical_scene_id": scene_id,
        "observer_id": observer_id,
        "name": str(scene.get("name", scene_id)),
        "description": description,
        "source_text": source_text,
        "active_layer_ids": [],
        "entity_visibility": {},
        "entity_overrides": {},
    }
    for layer in active_perception_layers(session, world, scene_id, observer_id):
        projection["active_layer_ids"].append(str(layer["id"]))
        if layer.get("name"):
            projection["name"] = str(layer["name"])
        layer_description = str(
            layer.get("description") or layer.get("desc") or "").strip()
        mode = str(layer.get("description_mode", "replace")).lower()
        if layer_description:
            if mode == "append" and projection["description"]:
                projection["description"] += "\n\n" + layer_description
            else:
                projection["description"] = layer_description
        layer_source = str(
            layer.get("source_text") or layer.get("source_quote") or "").strip()
        if layer_source:
            if mode == "append" and projection["source_text"]:
                projection["source_text"] += "\n\n" + layer_source
            else:
                projection["source_text"] = layer_source
        for entity_id in layer.get("hidden_entity_ids", []) or []:
            projection["entity_visibility"][str(entity_id)] = False
        for entity_id in layer.get("visible_entity_ids", []) or []:
            projection["entity_visibility"][str(entity_id)] = True
        overrides = layer.get("entity_overrides", {})
        if isinstance(overrides, dict):
            for entity_id, raw in overrides.items():
                if isinstance(raw, dict):
                    projection["entity_overrides"].setdefault(
                        str(entity_id), {}).update(deepcopy(raw))
    return projection


def entity_is_perceived(
    entity_id: str,
    session: dict,
    world: dict,
    base_visible: bool,
) -> bool:
    entity = world.get("entities", {}).get(entity_id, {})
    visible = False if entity.get("perception_only") is True else bool(base_visible)
    override = entity_visibility_override(entity_id, session, world)
    return visible if override is None else bool(override)


def entity_visibility_override(
    entity_id: str, session: dict, world: dict,
) -> bool | None:
    scene_id = str(session.get("player_state", {}).get("current_scene", ""))
    return scene_projection(session, world, scene_id)[
        "entity_visibility"].get(entity_id)


def projected_entity(
    entity_id: str, session: dict, world: dict, scene_id: str = "",
) -> dict:
    entity = deepcopy(world.get("entities", {}).get(entity_id, {}))
    scene_id = scene_id or str(
        session.get("player_state", {}).get("current_scene", ""))
    override = scene_projection(session, world, scene_id)[
        "entity_overrides"].get(entity_id, {})
    for key in (
        "name", "label", "public_label", "aliases", "desc", "description",
        "appearance", "interactions",
    ):
        if key in override:
            entity[key] = deepcopy(override[key])
    return entity


def render_scene_projection(session: dict, world: dict) -> str:
    projection = scene_projection(session, world)
    if not projection["active_layer_ids"]:
        return ""
    return (
        "=== ACTIVE OBSERVER SCENE PROJECTION (AUTHORITATIVE) ===\n"
        f"Physical scene: {projection['physical_scene_id']}\n"
        f"Observer: {projection['observer_id']}\n"
        f"Active perception layers: {', '.join(projection['active_layer_ids'])}\n"
        "Narrate only this projected description and projected entity visibility. "
        "The projection changes perception, never physical location or ownership.\n"
        f"Projected scene name: {projection['name']}\n"
        f"Projected description: {projection['description']}"
    )
