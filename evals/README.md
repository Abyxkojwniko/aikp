# AIKP playtests

These cases exercise grounding, clue gating, scene transitions, spoiler
resistance, and absent-entity handling against real modules kept outside the
repository.

The deterministic offline regression suite does not use an API key:

```bash
.venv/bin/python -m unittest discover -s tests -v
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
.venv/bin/python evals/run_manual_suite.py --fixtures-dir /home/lonpyer/aikp_eval_data/manual --output-dir /home/lonpyer/aikp_eval_runs
```

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

Object turns may use `"select_object": "entity_id"`. Expectations can assert
`inventory_contains`, `object_locations`, and `world_event_types_contains` so a
route cannot pass merely because its narration sounds correct while persisted
facts are wrong.

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
