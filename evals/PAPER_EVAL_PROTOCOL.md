# AIKP research evaluation protocol

Status: preregistration draft. The current gold set is preliminary and must receive an
independent second annotation before any paper claim is made.

## Research questions

1. Does hierarchical document mapping improve salient story-node and typed-relation
   recall over segmented extraction without increasing unsupported facts?
2. Does source-attributed node reconstruction improve local detail and object/location
   continuity while preserving the global causal graph?
3. Do deterministic runtime verifiers reduce hallucinated state transitions under
   adversarial player actions without preventing valid open-ended actions?

The design follows event-graph work that separates salient event discovery from graph
construction and explicitly repairs hallucinated or missing relations
([CALLMSAE, NAACL 2025](https://aclanthology.org/2025.naacl-long.112/)). It reports
node and typed-edge scores separately because complex narratives remain a known scene
decomposition bottleneck
([TSG-Bench, ACL 2025](https://aclanthology.org/2025.acl-long.1036/)).

## Corpus and splits

- `public-dev`: official zero-cost modules whose source files are downloaded separately;
  the repository stores only hashes, source URLs, and derived graph annotations.
- `private-test`: legally obtained modules not used while changing prompts or code.
- `localization-test`: Chinese community modules, reported separately from English
  publisher modules.
- `long-document-test`: quickstarts containing rules plus one or more adventures.

Report document characters/tokens, ruleset, language, number of independent scenarios,
gold nodes, branch edges, entities, physical scenes, and embedded narrative scopes. Do
not pool public-dev and private-test when selecting thresholds.

## Gold annotation

Two annotators independently mark:

- salient playable nodes and node kind;
- typed edges: `before`, `causes`, `enables`, `branches_to`, `reveals`, `pays_off`;
- canonical NPCs and unique objects with aliases;
- non-navigable places occurring only in books, dreams, memories, legends, examples,
  or backstory;
- branch terminal conditions and authoritative state changes.
- independent scenario membership for every node and entity in anthology documents;
  cross-scenario edges are forbidden in the playable graph.

Resolve disagreements only after calculating agreement. Report node agreement both with
and without the kind constraint, entity agreement by matched F1, and edge agreement by
typed-edge F1. Keep an adjudication log. Gold labels
must not quote module prose beyond names or short structural headings.
The exact decision rules are versioned in `ANNOTATION_GUIDE.md`. Validate both files
before agreement scoring; an annotation is not eligible for adjudication while it has
dangling edges, missing scenario membership, or a source-hash mismatch.

## Parser baselines and ablations

Run `heading_chain_v1` as a deterministic no-model lower bound. It promotes every
detected document heading to a playable node, assigns one scenario, emits only adjacent
`before` edges, and extracts no entities. Report it separately from the model-backed
conditions below; it is not a replacement for B0.

Report both all-document scenario assignment and `multi_scenario_assignment`. Treat the
latter as the primary anthology-boundary result: it excludes documents with only one
gold scenario, for which assigning every node to one bucket is trivially correct.

Run every condition with identical model, temperature, maximum output, and source text:

| ID | `AIKP_PARSER_ABLATION` | Purpose |
|---|---|---|
| B0 | `legacy` | segmented extraction baseline |
| B1 | `no_document_map` | removes long-document map/synthesis |
| B2 | `no_semantic_judge` | keeps deterministic gate only |
| B3 | `no_node_repair` | one node reconstruction attempt |
| Ours | `full` | complete system |

Use at least three parser seeds or provider repetitions per condition. Report mean,
standard deviation, and paired bootstrap 95% confidence intervals by module. Preserve
raw predictions and model/provider identifiers alongside reports.

`run_parser_matrix.py` treats a run as invalid if any provider request fails, even when
the parser can produce a fallback world book. It will not score an incomplete repeat.
Successful predictions are content-hashed, and an existing experiment directory cannot
be resumed with a different model, source root, condition set, or module set.
The manifest contents and executable experiment source tree are also hashed, so a dirty
worktree cannot silently resume artifacts produced by different code. Provider
repetitions are indexed but are not described as seeded unless the provider actually
supports and receives a seed.
The adjudicated gold directory is content-hashed into experiment identity. Conditions
are deterministically shuffled within each `(repeat, module)` block to reduce temporal
provider drift; the ordering seed and complete order hash are archived.

For comparisons, average repetitions within each module first, then paired-bootstrap
modules as independent clusters. Do not bootstrap individual runs as if three outputs
from one source document were three independent modules. Use a fixed published bootstrap
seed and report the number of modules and paired runs.

## Structural metrics

Run `evals/story_graph_benchmark.py` without an API key. Primary metrics are macro and
micro node F1, kind-constrained typed-node F1, and typed-edge F1. Secondary metrics are
typed entity F1, non-navigable
scope accuracy, cross-scenario edge isolation, node/entity scenario-assignment coverage,
scenario pairwise-clustering F1, detailed-record provenance coverage, and graph closure.
Generated scenario ids are permutation-invariant: clustering is scored by whether matched
node/entity pairs belong together, not by exact id strings. Missing memberships are
unassigned singletons and cannot receive a perfect isolation score.

Exact string match is not sufficient for generative identifiers, so matching uses a
fixed deterministic alias/token similarity threshold before scoring. Freeze this
threshold on `public-dev`; do not tune it on `private-test`.

Grounding is a separate axis from fluency. This mirrors FACTS Grounding's separation of
instruction fulfillment and factual support against a supplied long document
([FACTS Grounding](https://arxiv.org/abs/2501.03200)). Any semantic judge used for
atomic-claim support must be calibrated against human labels and reported separately
from deterministic source-quote verification.

## Interactive evaluation

Each gold graph produces successful, failure, refusal, backtracking, guessed-secret,
same-name-object, NPC-death, inventory-transfer, and illegal-scene-jump trajectories.
The authoritative final session state, not narrator wording, determines success. This
follows the final-state evaluation principle used by
[$\tau$-bench](https://arxiv.org/abs/2406.12045).

Report:

- task success and branch coverage;
- unsupported world mutations per 100 turns;
- hidden-information leaks per 100 turns;
- valid-action false rejection rate;
- object/NPC location continuity violations;
- `pass^k` for repeated trajectories, with `k` declared in advance. A case must
  have at least `k` independent transcripts or its `pass^k` is reported as null;
  tables report distinct cases and total runs separately;
- latency, parser calls, narrator calls, and token/cost totals.

Expected action validity and outcome are authored before a run. The runner independently
records whether the deterministic verifier blocked the request or the narrator accepted
it. Never copy an expected outcome into the observed field. Hallucination-risk labels are
post-run human observations; missing labels are excluded and reported as `null`, never as
zero violations over an invented denominator.

## Reproduction commands

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python evals/validate_gold.py \
  --gold-dir evals/gold --source-root /home/lonpyer/aikp_eval_data
.venv/bin/python evals/annotation_packets.py prepare \
  --source-root /home/lonpyer/aikp_eval_data \
  --output-dir /home/lonpyer/aikp_annotation_packets/v1 \
  --annotator annotator-a --annotator annotator-b
.venv/bin/python evals/annotation_agreement.py \
  --annotator-a /path/to/collected-a/annotations \
  --annotator-b /path/to/collected-b/annotations \
  --source-root /home/lonpyer/aikp_eval_data
.venv/bin/python evals/story_graph_benchmark.py \
  --predictions-dir /path/to/predictions/full \
  --source-root /home/lonpyer/aikp_eval_data \
  --output /path/to/reports/full.json
.venv/bin/python evals/run_heading_baseline.py \
  --source-root /home/lonpyer/aikp_eval_data \
  --output-dir /path/to/heading-chain-v1
.venv/bin/python evals/run_parser_matrix.py \
  --output-dir /path/to/parser-matrix --repeats 3 --dry-run
# For an actual run, set AIKP_EVAL_API_KEY explicitly and remove --dry-run.
.venv/bin/python evals/compare_parser_conditions.py \
  --experiment-dir /path/to/parser-matrix \
  --reference legacy --compared full \
  --bootstrap-samples 10000 --seed 20260818
.venv/bin/python evals/run_manual_suite.py \
  --fixtures-dir /home/lonpyer/aikp_eval_data/manual \
  --output-dir /home/lonpyer/aikp_eval_runs
.venv/bin/python evals/validate_interactive_cases.py \
  --fixtures-dir /home/lonpyer/aikp_eval_data/manual --require-annotations
.venv/bin/python evals/interactive_benchmark.py \
  --runs-dir /home/lonpyer/aikp_eval_runs --latest-per-case \
  --coverage-dir /home/lonpyer/aikp_eval_data/manual --pass-k 3
.venv/bin/python evals/build_paper_tables.py \
  --interactive-report /home/lonpyer/aikp_eval_runs/benchmark.json \
  --parser-baseline /path/to/heading-chain-v1/benchmark.json \
  --output-json /path/to/paper-results.json \
  --output-markdown /path/to/paper-results.md
```

Before release, pin Python dependencies, record source hashes, archive all prediction
JSON, publish the annotation guide and adjudication log, and rerun the full matrix from
a clean checkout.
