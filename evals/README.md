# AIKP playtests

These cases exercise grounding, clue gating, scene transitions, spoiler
resistance, and absent-entity handling against real modules kept outside the
repository.

The deterministic offline regression suite does not use an API key:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Story graph benchmark

`gold/` stores derived story-node, typed-edge, entity, and narrative-scope annotations;
it does not contain module prose. Download the source modules from the publisher URLs
recorded in each gold JSON, then preserve the declared filename under an external source
root. Score parser predictions without an API key:

```bash
.venv/bin/python evals/story_graph_benchmark.py \
  --predictions-dir /path/to/world-books \
  --source-root /home/lonpyer/aikp_eval_data \
  --output /path/to/report.json
```

Prediction filenames must be `<module_id>.json`. The report separates node F1,
kind-constrained typed-node F1, typed-edge F1, typed entity F1,
embedded-scope false positives, cross-scenario
edge isolation, permutation-invariant node/entity scenario assignment, source
provenance, and graph closure. See `PAPER_EVAL_PROTOCOL.md` for
splits, annotation, ablations, repeated runs, and interactive metrics.
The aggregate also reports `multi_scenario_assignment`, restricted to documents with
more than one gold scenario; this is the primary anthology-boundary metric because
single-scenario documents make one-bucket assignment trivially correct.
Multi-adventure predictions retain a scenario registry, and runtime scene/entity ids
are namespaced so repeated local names cannot collide in the flat world book.

Run the no-model heading-chain lower bound before any provider experiment. It treats
every detected heading as playable, assumes one scenario, and connects adjacent
sections with `before`, so its limitations are explicit and reproducible:

```bash
.venv/bin/python evals/run_heading_baseline.py \
  --source-root /home/lonpyer/aikp_eval_data \
  --output-dir /home/lonpyer/aikp_eval_runs/heading-chain-v1
```

Validate gold structure and compare two frozen independent annotations before
adjudicating them:

```bash
.venv/bin/python evals/validate_gold.py \
  --gold-dir evals/gold --source-root /home/lonpyer/aikp_eval_data
.venv/bin/python evals/annotation_agreement.py \
  --annotator-a /path/to/annotator-a --annotator-b /path/to/annotator-b \
  --source-root /home/lonpyer/aikp_eval_data
```

`ANNOTATION_GUIDE.md` fixes node granularity, edge semantics, scenario boundaries,
embedded-story handling, and object-location rules.

Prepare two isolated blind annotation packets without exposing existing gold or parser
predictions. Source copies are opt-in so publisher material is not redistributed by
accident. After each annotator completes their own directory, collect and freeze the
submissions before agreement scoring:

```bash
.venv/bin/python evals/annotation_packets.py prepare \
  --source-root /home/lonpyer/aikp_eval_data \
  --output-dir /home/lonpyer/aikp_annotation_packets/v1 \
  --annotator annotator-a --annotator annotator-b
.venv/bin/python evals/annotation_packets.py collect \
  --packet /home/lonpyer/aikp_annotation_packets/v1/coordinator_packet.json \
  --annotator annotator-a \
  --annotation-dir /home/lonpyer/aikp_annotation_packets/v1/annotator-a/annotations \
  --output-dir /home/lonpyer/aikp_annotation_submissions/annotator-a
```

Distribute only the individual annotator directory, never `coordinator_packet.json`.
Run agreement on each collected `annotations/` subdirectory.

Plan the complete five-condition, three-repeat parser experiment without making API
calls, then run it only with a separately supplied evaluation credential:

```bash
.venv/bin/python evals/run_parser_matrix.py \
  --output-dir /path/to/parser-matrix --repeats 3 --dry-run
AIKP_EVAL_API_KEY=... .venv/bin/python evals/run_parser_matrix.py \
  --output-dir /path/to/parser-matrix --repeats 3
.venv/bin/python evals/compare_parser_conditions.py \
  --experiment-dir /path/to/parser-matrix \
  --reference legacy --compared full
```

The matrix runner never reads `DEEPSEEK_API_KEY` as its experiment credential. It
stores raw predictions, source/prediction hashes, parser mode, validation count,
elapsed time, provider-call failures, characters, and token usage for each run. A
provider failure invalidates the whole repeat instead of silently scoring fallback
output. It locks the gold-tree hash and deterministically interleaves conditions within
repeat/module blocks to limit provider drift. Condition comparison uses
module-clustered paired bootstrap intervals.

Audit every external source hash, ruleset classification, structural segmentation, and
long-document window coverage without making model calls:

```bash
.venv/bin/python evals/audit_corpus.py \
  --source-root /home/lonpyer/aikp_eval_data \
  --strict
```

`run_playtest.py` is optional end-to-end tooling for evaluating a configured
runtime model. It is not needed for the offline suite. To use it explicitly:

```bash
.venv/bin/python evals/run_playtest.py \
  --case evals/cases/lightless_beacon_adversarial.json
```

The first run parses the module. Later runs can add `--reuse-world`. Transcripts
and the exact per-turn model context are written under
`/home/lonpyer/aikp_eval_runs` and are not committed.

For no-key evaluation, `run_manual_playtest.py` injects recorded narration into
the normal engine while preserving redaction, movement, dice, trust, logging,
and persistence. It also records every exact prompt and checks expectations:

```bash
.venv/bin/python evals/run_manual_playtest.py \
  --world /path/to/world.json \
  --case /path/to/case.json \
  --responses /path/to/responses.json
```

This mode disables auxiliary LLM enrichment and never creates an API client.

Run every `*_world.json`, `*_case.json`, and `*_responses.json` fixture triple
in a directory:

```bash
.venv/bin/python evals/run_manual_suite.py \
  --fixtures-dir /home/lonpyer/aikp_eval_data/manual
```

A coverage manifest named `<module>_coverage.json` makes the suite fail unless
every declared source segment, branch outcome, lifecycle event, and ending was
covered by a successful case. Regenerate the exhaustive fixtures for the three
local modules and the adversarial fixtures for five additional official free
adventures, then run all routes with:

```bash
.venv/bin/python evals/generate_exhaustive_fixtures.py
.venv/bin/python evals/generate_expanded_fixtures.py
.venv/bin/python evals/generate_identity_stress_fixtures.py
.venv/bin/python evals/generate_coriolis_fixtures.py
.venv/bin/python evals/generate_starfinder_fixtures.py
.venv/bin/python evals/generate_pendragon_fixtures.py
.venv/bin/python evals/generate_haunting_fixtures.py
.venv/bin/python evals/generate_scritch_scratch_fixtures.py
.venv/bin/python evals/run_manual_suite.py --fixtures-dir /home/lonpyer/aikp_eval_data/manual --output-dir /home/lonpyer/aikp_eval_runs
```

Validate that every paper-evaluation turn has a preregistered validity and expected
outcome, then aggregate independently observed verifier/narrator outcomes:

```bash
.venv/bin/python evals/validate_interactive_cases.py \
  --fixtures-dir /home/lonpyer/aikp_eval_data/manual --require-annotations
.venv/bin/python evals/interactive_benchmark.py \
  --runs-dir /home/lonpyer/aikp_eval_runs --latest-per-case \
  --coverage-dir /home/lonpyer/aikp_eval_data/manual --pass-k 3
.venv/bin/python evals/build_paper_tables.py \
  --interactive-report /home/lonpyer/aikp_eval_runs/benchmark.json \
  --parser-baseline /home/lonpyer/aikp_eval_runs/heading-chain-v1/benchmark.json \
  --output-json /path/to/paper-results.json \
  --output-markdown /path/to/paper-results.md
```

Legacy hand-authored cases receive versioned annotations from
`interactive_annotations.json`; generated cases embed the same schema. Risk metrics with
no post-run human observations are `null`, not a claimed zero hallucination rate.
The aggregate reports distinct cases separately from transcript runs. Per-case
`pass^k` is emitted only when that case has at least `k` independent transcripts;
otherwise the value is `null` and `sufficient_runs` is false.
The synthetic identity/scope fixture targets same-name doors, embedded-story movement,
hidden-item leakage, cross-scene inventory, backtracking, item drops, remote-item use,
and dead-NPC dialogue in one persisted session.

The source-grounded Starfinder fixture adds route phrases such as `take the stairs`,
numbered location headings, forged hidden-object selection, non-adjacent teleportation,
reactor/launcher repair branches, two distinct doors, inventory transfer, and defeated,
dead, or fled NPC dialogue. Its world is derived from Paizo's freely downloadable
*Battle for Nova Rush* and the source hash is locked in `corpus_manifest.json`.

The source-grounded Pendragon fixture reconstructs Chaosium's freely available
*The Sword Tournament*. Its three routes cover the city clues, tournament opponent
branches, early/late cathedral arrival, intervention and loyalty choices, guessed
hidden objects, forged NPC selection, non-physical biography locations, and the
authoritative movement of Arthur and the Sword in the Stone across scenes.

The source-grounded Call of Cthulhu fixture reconstructs Chaosium's freely available
*The Haunting*. Its research, false-report, and loss routes cover condition-gated
archives and destinations, containers that disclose multiple entities, historical
name disclosure without moving an NPC, hidden and carried objects, authored trigger
prerequisites, non-physical delusion scopes, dormant/dead NPC interaction, and both
successful and destructive conclusions. Authored `on_trigger.events` are committed to
the immutable world-event log, while `requires_inventory`, `requires_flags`, and
`requires_entity_states` are checked in code before narration can execute the branch.

The *Scritch Scratch* fixture is a true investigation sandbox rather than a linear
chapter chain. Its evidence gate requires the cottage and church findings plus either
the pub or museum history branch. The routes exercise inventory- and state-gated
exits, premature-climax refusal, locked-door bypass attempts, embedded Bible and
diorama settings, hidden chemicals, an NPC who becomes non-interactable during the
same dialogue turn, and both fire and flight conclusions. Exit prerequisites use the
same closed-world inventory, all/any flag, and entity-state vocabulary as authored
object actions.

Ruleset detection and automatic check resolution are separate capabilities. Coriolis:
The Great Dark is identified as a `d6_pool` system for parsing and evaluation, but its
automatic adapter remains disabled until Base/Gear Dice, pushing, and Hope loss are
implemented together. The runtime must not silently substitute a d20 check.

The expanded source PDFs are kept outside the repository in
`/home/lonpyer/aikp_eval_data/expanded_modules`. They are Chaosium's free
adventures: *Alone Against the Flames*, *The Great Hunt*, *Quest of the Red
Blade*, *The Sword of Kings*, and *The Rattling Wind*. The generated cases
exercise refusal, backtracking, direct scene skips, guessed hidden objects,
NPC selection and death, split-party choices, combat and non-combat solutions,
false accusations, time-wasting, capture, failure, alternate endings, invented
object rejection, cross-scene inventory, and validated object use.

A case turn may set authoritative entity lifecycle state before input, for
example `"set_entity_states": {"innkeeper": "dead"}`. This proves that stale
NPC selections are cleared and dialogue is blocked without consuming a
narration response.
World authors may also set `states.<state>.interactable` to `false`; this supports
module-specific states such as escaped or removed without incorrectly treating every
generic `defeated` state as dead.

Object turns may use `"select_object": "entity_id"`. Expectations can assert
`inventory_contains`, `object_locations`, and `world_event_types_contains` so a
route cannot pass merely because its narration sounds correct while persisted
facts are wrong.

Source-authored `scene.entry_events` may commit one-shot `entity_moved` events when
a recurring person or object canonically enters a later scene. The runtime validates
the referenced entity and scene ids, records the event in the immutable world-event
log, and does not replay the move when players backtrack and re-enter the scene.

Movement turns may use `"select_scene": "scene_id"`. The selected id must be a
currently reachable, player-visible exit. This lets a route use pronouns such as
"go there" without allowing the narrator to choose a different or hidden scene.

Use `"roll_verdict": "extreme_success"` on a turn to resolve its pending CoC
check with a deterministic result. Expectations are evaluated after the roll,
so `"expect": {"clocks": {"development_round": 4}}` verifies the persisted
clock rather than merely checking narration text.

Manual runs remove their temporary world book and session after writing the
transcript. Add `--keep-runtime-state` only when inspecting a failed run in the
live application.

Reparse the three modules in `/home/lonpyer/下载` without an API client, then
run hostile-narrator tests against the generated, app-loadable worlds:

```bash
.venv/bin/python evals/reparse_local_modules.py
.venv/bin/python evals/run_destructive_reparse_test.py
```

The reparse report records source hashes, bound and unbound scenes, recovered
source markers, and unclassified structural candidates. The destructive test
injects forged destruction, mass death, invented inventory, illegal movement,
forged movement markers, and selected-object deletion while comparing the full
authoritative state before and after every turn. It disables narration,
planning, enrichment, and summary API calls; no configured key is used.
