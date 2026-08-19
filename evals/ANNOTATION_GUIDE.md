# AIKP annotation guide

Version: 0.1. Annotators work independently and must not inspect parser output.

## General rules

- Read the complete adventure once before labeling. Record its global premise,
  independent scenarios, terminal conditions, and recurring entities first.
- On a second pass, label detailed nodes and relations while keeping the global map
  visible. Do not infer events, exits, identities, or outcomes absent from the source.
- Use short paraphrases. Source quotes are limited to proper names and short headings.
- Mark uncertainty in annotator notes; do not silently resolve it from outside lore.
- IDs are annotation-local. Agreement and benchmark matching are permutation-invariant.

## Scenario boundary

A scenario is an independently playable adventure with its own entry condition and at
least one terminal outcome. Rules examples, fictional anecdotes, dreams, memories,
legends, and books found inside an adventure are not independent scenarios. Anthology
chapters are separate scenarios even when they reuse a rules chapter or setting.

Every node and entity receives exactly one `scenario_id`. A genuinely shared rules
entity is omitted from the playable entity set rather than attached to every scenario.
Playable graph edges may never cross scenario boundaries.

## Story nodes

Create one node for a player-relevant state or event that changes at least one of:

- available actions or reachable places;
- known clues or interpretation of prior evidence;
- NPC/object lifecycle, ownership, or location;
- branch conditions, stakes, or terminal outcome.

Split a paragraph into multiple nodes when those changes can occur independently. Merge
adjacent prose when it describes one indivisible game state. Exclude atmosphere,
rules examples, repeated reminders, and purely stylistic narration.

Allowed node kinds are `opening`, `event`, `choice`, `clue`, `encounter`, and `ending`.
Use `choice` only when player agency selects among materially different continuations.

## Typed edges

- `before`: source ordering is required, but no stronger dependency is stated.
- `causes`: the source event directly produces the target event or state.
- `enables`: the source makes the target possible without directly causing it.
- `branches_to`: a choice or check can select the target continuation.
- `reveals`: the source discloses the target clue, identity, or interpretation.
- `pays_off`: the target resolves or materially reuses setup from the source.

Prefer the strongest supported relation. Do not add transitive edges. Every endpoint
must be a labeled node in the same scenario.

## Scenes and embedded places

A navigable scene is a physical or intentionally traversable mental location where a
player can act. Two doors, rooms, or streets with the same common name remain distinct
if their containing location or reachable entities differ.

A place mentioned only inside a storybook, dream report, memory, legend, backstory, or
rules example is a `narrative_scope`, not a runtime scene. If play explicitly enters a
dream or memory and permits choices there, it becomes navigable and is labeled as such.

## Entities and objects

Label named NPCs and unique player-relevant objects. Aliases include titles, surnames,
nicknames, translations, and unambiguous descriptive references supported by the text.
Do not merge two entities merely because they share a name or role.

Object location is authoritative state: distinguish scene location, another entity's
possession, player inventory, destroyed/consumed state, and hidden/unrevealed state.
An object cannot appear in another scene unless a labeled action or event moves it.
Corpses remain inspectable objects but dead NPCs are not valid dialogue targets.

## Interactive trajectories

For each player action, annotate `action_validity` as `valid`, `invalid`, or
`ambiguous`, plus `expected_outcome` as `accepted` or `blocked`. Validity is judged from
the authoritative state before the action, not narrator plausibility.

Risk labels are written only to the transcript's `observed` object after seeing the
response and final state; they must never appear in the preregistered `evaluation`:

- `unsupported_world_mutation`: authoritative state changed without source/action support;
- `hidden_information_leak`: response exposed information not yet available to players;
- `location_continuity_violation`: NPC/object appeared, moved, or acted from an invalid place.

Do not fill absent observations with `false`; absent means unannotated and is excluded
from denominators. The runner records observed acceptance independently from the expected
label, so valid-action false rejection and invalid-action acceptance remain measurable.

## Agreement and adjudication

Freeze both independent files before running `annotation_agreement.py`. Adjudication is
a third artifact containing module, item type, both labels, decision, reason, and
adjudicator. Never overwrite the two original annotations. Report agreement before
adjudication and benchmark scores against only the adjudicated gold.
The adjudication artifact is a JSON array conforming to
`adjudication_log.schema.json`; `item_key` is a stable local description, not either
annotator's generated ID.
