# C3 Execution Plan
## Phase-by-Phase Build Prompts for Codex

**Companion to:** `docs/C3_SYSTEM_SPEC.md` (the architecture anchor — read it first).
**Audience:** OpenAI Codex, building Component 3 incrementally into the existing repository.

**Golden rules for every phase:**
- Read `docs/C3_SYSTEM_SPEC.md` before writing code. It is the single source of truth for architecture, schema, paths, and contracts.
- Do **not** modify any C1 or C2 source (`src/c1_detector/`, `src/c2_synthesis/`, `src/data/`) or anything under `data/`. C3 consumes them read-only.
- **Every** `torch.utils.data.DataLoader` uses `num_workers=0` (Windows spawn-deadlock protection).
- All hyperparameters come from YAML configs — no magic numbers in code.
- The base VLM and tokenizer are cached locally; never re-download inside a training loop.
- Global deterministic seeding via `utils/seed.py` in every entry point.
- A phase is complete only when its `tests/verify_phase_X.py` **exits 0**. Do not proceed otherwise.
- After each artifact-producing phase, stage DVC/git as specified in the spec.
- If any instruction is ambiguous or conflicts with the repo, **stop and ask** rather than guessing.

**Note on phase-test filenames:** C1 and C2 already created `tests/verify_phase_*.py`. To avoid collisions, C3 phase tests are named `tests/verify_c3_phase_X.py`. Use that exact pattern.

---

## Phase 0 — Environment, Scaffolding, and Quantised-Model Sanity

**Objective:** Stand up the C3 package skeleton and prove the quantised Qwen2.5-VL model loads and generates on the Blackwell under Windows. This phase de-risks the single biggest technical threat: 4-bit `bitsandbytes` on Windows.

**Files to Create / Modify:**
- `src/c3_explanation/__init__.py`
- `src/c3_explanation/utils/__init__.py`
- `src/c3_explanation/utils/seed.py`
- `src/c3_explanation/utils/logging_utils.py`
- `src/c3_explanation/utils/json_io.py`
- `src/c3_explanation/utils/mlflow_utils.py`
- `src/c3_explanation/configs/qwen_vl_base.yaml`
- `src/c3_explanation/model/__init__.py`
- `src/c3_explanation/model/load_vlm.py`
- `requirements_c3.txt` (additive; do not touch existing requirements)
- `tests/verify_c3_phase_0.py`

**Detailed Instructions & Logic:**
1. Create the full `src/c3_explanation/` package tree from the spec (all `__init__.py`, empty sub-packages allowed for now).
2. `requirements_c3.txt` pins: `transformers`, `peft`, `bitsandbytes`, `accelerate`, `qwen-vl-utils`, `pyyaml`, `jsonschema`, `mlflow`, `Pillow`. Match versions in the env where possible; do not downgrade torch.
3. `utils/seed.py`: `set_global_seed(seed: int)` seeding `random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`.
4. `utils/logging_utils.py`: `get_logger(name)` → console+file logger to `outputs/logs/c3/`.
5. `utils/json_io.py`: `load_json(path)`, `save_json(path, obj)`, and `validate_against_schema(obj, schema_path)` using `jsonschema`.
6. `utils/mlflow_utils.py`: `start_c3_run(experiment, run_name, params)` wrapping the existing store at `file:./outputs/logs/mlflow`.
7. `configs/qwen_vl_base.yaml`: `model_id: Qwen/Qwen2.5-VL-7B-Instruct`, `quantization: nf4`, `compute_dtype: bfloat16`, `local_cache_dir`, generation params (`max_new_tokens`, `temperature: 0.0`, `do_sample: false`), and a `fallback` block (`quantization_fallback: [nf4, int8, bf16]`, `model_id_fallback: Qwen/Qwen2.5-VL-3B-Instruct`).
8. `model/load_vlm.py`:
   - `load_quantised_vlm(config) -> (model, processor)` loading the model in 4-bit NF4 via `BitsAndBytesConfig`, on CUDA, cached locally.
   - It must attempt the fallback chain from config if 4-bit load fails, logging which precision succeeded.

**Integration requirements:** MLflow store path must match C1/C2 (`file:./outputs/logs/mlflow`). Do not create a second store.

**Edge Case & Windows Protections:**
- Wrap the 4-bit load in a try/except that walks the fallback chain (nf4 → int8 → bf16 3B) and logs the outcome. Do not crash on a `bitsandbytes` import/runtime failure — record it and fall back.
- No DataLoaders yet; if any appear later, `num_workers=0`.
- Local caching: set the HF cache to the configured local dir so no repeated downloads.

**Verification Script & Acceptance Test:** `tests/verify_c3_phase_0.py`
- Imports all new util modules.
- Calls `load_quantised_vlm` and asserts a model + processor are returned and report which precision loaded.
- Runs one generation on a small test image (a synthetic 448×448 RGB array) with a trivial prompt ("Describe this image in one sentence.") and asserts non-empty text output with no exception.
- Prints GPU name, the precision that loaded, peak VRAM, and elapsed time.
- Exits 0 on success.

---

## Phase 1 — Grounding Bridge, Severity Rules, and Report Schema

**Objective:** Build the deterministic, parameter-free bridge that converts a C1 mask into structured facts and a schema-valid skeleton report. This is the highest-leverage component in C3 and involves no model.

**Files to Create / Modify:**
- `src/c3_explanation/configs/report_schema.yaml`
- `src/c3_explanation/grounding/__init__.py`
- `src/c3_explanation/grounding/bridge.py`
- `src/c3_explanation/grounding/severity_rules.py`
- `tests/verify_c3_phase_1.py`

**Detailed Instructions & Logic:**
1. `report_schema.yaml`: the exact JSON schema from spec §3.2 as a `jsonschema`-compatible definition, plus a `severity_thresholds` block (area-percentage cutoffs per severity level, optionally per defect-type) and the allowed enumerations (separate `defect_type` lists per dataset whose union is used by the common report schema, `recommended_action.action` enum, `severity.level` enum).
2. `severity_rules.py`:
   - `assign_severity(area_pct: float, defect_type: str, config) -> str` returning `minor|moderate|severe` by the documented thresholds. Pure function, fully determined by config.
3. `bridge.py`:
   - `ground_mask(mask: np.ndarray, image_shape, defect_type, config) -> dict` returning the structured facts: `bounding_box [x,y,w,h]`, `centroid [cx,cy]`, `area_pct`, `region` (quadrant/edge label from the centroid), and `severity_level` (via `assign_severity`).
   - `build_skeleton_report(facts: dict, defect_type: str) -> dict` returning a schema-valid report with the bridge-owned factual fields populated and the model-owned language fields left as empty strings/placeholders. `recommended_action.action = "inspect"` is an internal Phase 1 schema-valid placeholder only, not a Bridge inference; Phase 2 corpus construction and Phase 4 final generation must replace/populate it.
   - `region` labelling: divide the frame into a 3×3 grid (or quadrants) and map the centroid to a plain-language label ("upper-left", "centre", "right edge", etc.). Deterministic.

**Integration requirements:** Reads masks via the project mask contract (binary, >127). Uses `utils/json_io.validate_against_schema`. No model, no C1 re-run — operates on a mask array passed in.

**Edge Case & Windows Protections:**
- Empty/all-zero mask → `defect_present: false`; `bounding_box`, `centroid`, and `region` are `null`; `affected_area_pct: 0.0`; and `severity_level: none`. The `none` level is exclusive to this no-defect case; handled without error.
- Multi-blob masks → bounding box of the union, centroid of the largest connected component (documented choice).
- Area percentage computed against full image area, clamped to [0, 100].

**Verification Script & Acceptance Test:** `tests/verify_c3_phase_1.py`
- Constructs several synthetic masks (a small corner blob, a centred blob, a near-full-frame mask, an empty mask) and asserts:
  - bounding boxes and centroids are numerically correct against direct computation,
  - `area_pct` matches the hand-computed fraction,
  - `region` labels match the blob positions,
  - `assign_severity` returns the expected level for known area/type inputs,
  - `build_skeleton_report` output passes `validate_against_schema`.
- Runs the bridge on at least one real C1 mask from `data/processed/` and asserts a schema-valid skeleton is produced.
- Exits 0 on success.

---

## Phase 2 — Report Corpus Construction

**Objective:** Build the templated ground-truth report corpus for all real MVTec and ECF defects, split train/test, with C2 synthetic defects held out.

**Files to Create / Modify:**
- `src/c3_explanation/configs/datasets/mvtec.yaml`
- `src/c3_explanation/configs/datasets/ecf.yaml`
- `src/c3_explanation/data/__init__.py`
- `src/c3_explanation/data/report_builder.py`
- `src/c3_explanation/data/enrichment.py`
- `src/c3_explanation/data/corpus_split.py`
- `tests/verify_c3_phase_2.py`

**Detailed Instructions & Logic:**
1. Dataset YAMLs: class lists, the `dataset_key`/`processed_dir`/`splits_path` mapping (ECF → `ECF-Dataset`), and a `defect_type -> {description_template, action_template}` mapping per class. Exclude the same ECF classes excluded elsewhere where they have no usable masks. Corpus construction must enforce the applicable dataset-specific defect-type vocabulary.
2. `report_builder.py`:
   - `build_report(image_path, mask_path, class_name, defect_type, config) -> dict` = run the bridge for facts, apply templates for `description`/`recommended_action`, replace the Phase 1 skeleton action placeholder, and assemble a full schema-valid report. Deterministic.
   - `build_corpus(dataset, config)` iterates the manifest's defective images and writes `(image reference, grounding.json, report.json)` triples per the spec §3.3 contract.
3. `enrichment.py` (optional): `enrich_description(report, ...)` rephrases only the `description` field, preserving all factual fields; gated behind a config flag, default off. If it uses a model, that call is offline and clearly separated from inference.
4. `corpus_split.py`: deterministic seeded train/test split over real defects; assert **no C2 synthetic paths** enter the corpus; write split membership.

**Integration requirements:** Uses `grounding/bridge.py` and the split manifests. Reads images/masks read-only. Writes to `outputs/corpus/c3/`.

**Edge Case & Windows Protections:**
- Defects with null/missing masks → skipped with a logged warning (mirrors ECF handling elsewhere).
- Every produced `report.json` must pass schema validation at write time; a failing report aborts with a clear message rather than being written.
- POSIX paths in all written JSON.

**Verification Script & Acceptance Test:** `tests/verify_c3_phase_2.py`
- Builds the corpus for one MVTec class and one ECF class.
- Asserts every written `report.json` passes `validate_against_schema`.
- Asserts every `grounding.json` matches its report's factual fields.
- Asserts the train/test split is deterministic (re-running yields identical membership) and contains **no** paths under any C2 synthetic directory.
- Prints per-class corpus counts.
- Exits 0 on success.

---

## Phase 3 — QLoRA Fine-Tuning

**Objective:** Fine-tune Qwen2.5-VL-7B with QLoRA on the report corpus so it emits schema-valid reports that respect the grounded facts.

**Files to Create / Modify:**
- `src/c3_explanation/configs/qlora_report.yaml`
- `src/c3_explanation/model/train_report_qlora.py`
- `tests/verify_c3_phase_3.py`

**Detailed Instructions & Logic:**
1. `qlora_report.yaml`: `rank: 32`, `alpha: 32`, `lr: 1e-4`, `max_steps` (start modest), target modules (LM attention projections), `batch_size: 1`, `gradient_accumulation`, `seed: 42`, prompt template that injects the grounded facts as text alongside the image.
2. `train_report_qlora.py`:
   - Loads the quantised base via `load_vlm.load_quantised_vlm`, attaches QLoRA adapters via `peft`.
   - Builds a dataset from the Phase 2 corpus: each example is (image, grounded-facts-prompt) → target report JSON string. `num_workers=0`.
   - Trains with the standard causal-LM loss on the target report tokens; logs loss to MLflow.
   - Saves the adapter + tokenizer to `outputs/checkpoints/c3/{dataset}/`.
   - **Timed smoke test** (a few steps) before the full run: print steps/sec and projected wall time; abort if implausible.

**Integration requirements:** Consumes Phase 2 corpus; uses `seed`, `mlflow_utils`, `load_vlm`. Writes to the spec checkpoint path.

**Edge Case & Windows Protections:**
- `num_workers=0` on the training DataLoader — asserted in the test.
- bf16 adapter compute; loss in fp32 for stability.
- Timed smoke test mandatory before the full fine-tune.
- If the 4-bit backend was unavailable in Phase 0 and a fallback precision is in use, honour it here (read the recorded precision, do not re-attempt 4-bit blindly).

**Verification Script & Acceptance Test:** `tests/verify_c3_phase_3.py`
- Runs a short (e.g. 20-step) fine-tune on a small slice of one dataset's corpus.
- Asserts the loss is finite and decreases (mean of first few vs last few steps).
- Asserts the adapter + tokenizer are written and reloadable.
- Loads the adapter and generates one report for a held-out image; asserts it parses as JSON and passes `validate_against_schema`.
- Asserts the training DataLoader used `num_workers=0`.
- Prints steps/sec and projected full-run wall time.
- Exits 0 on success.

---

## Phase 4 — Generation and the Four Metrics

**Objective:** Generate reports for the held-out real-defect test set with the fine-tuned model, and implement the four evaluation axes.

**Files to Create / Modify:**
- `src/c3_explanation/model/generate_report.py`
- `src/c3_explanation/metrics/__init__.py`
- `src/c3_explanation/metrics/schema_compliance.py`
- `src/c3_explanation/metrics/factual_accuracy.py`
- `src/c3_explanation/metrics/hallucination.py`
- `src/c3_explanation/metrics/report_quality.py`
- `src/c3_explanation/metrics/report.py`
- `tests/verify_c3_phase_4.py`

**Detailed Instructions & Logic:**
1. `generate_report.py`:
   - `generate(image_path, grounding_facts, model, processor, config) -> dict` = build the grounded prompt, run the VLM, parse the output to JSON, populate rather than inherit the Phase 1 skeleton action placeholder, enforce the applicable dataset-specific defect-type vocabulary, and enforce the bridge-owned factual fields (overwrite any model drift in location/area/severity_level with the deterministic bridge values). Writes report + meta per spec §3.4.
   - Deterministic decoding (temperature 0 / greedy) for reproducibility.
2. `schema_compliance.py`: fraction of reports passing `validate_against_schema`.
3. `factual_accuracy.py`: `defect_type` accuracy; location IoU/centroid distance vs ground-truth mask; `affected_area_pct` absolute error.
4. `hallucination.py`: automatic contradiction checks (does any stated fact contradict a grounded field?) plus hooks for a rubric on free-text fields; returns a per-report hallucination flag and an aggregate rate.
5. `report_quality.py`: rubric scoring interface (human-sample or LLM-as-judge); returns a score per report.
6. `report.py`: assemble `outputs/tables/c3/{compliance,accuracy,hallucination,quality}.csv`.

**Integration requirements:** Uses Phase 3 adapter, Phase 1 bridge, Phase 2 test split. Logs metrics to MLflow.

**Edge Case & Windows Protections:**
- Unparseable model output → counts as a schema-compliance failure, does not crash the run.
- The factual-field enforcement step guarantees location/area cannot hallucinate even if the model drifts — assert this in the test.
- `num_workers=0` anywhere batching is used.

**Verification Script & Acceptance Test:** `tests/verify_c3_phase_4.py`
- Generates reports for a small held-out slice using the Phase 3 adapter.
- Asserts every metric CSV is written with finite values.
- Asserts the factual-field enforcement: for a report where the model was made to output a wrong bounding box, the final written report still carries the bridge's correct box.
- Asserts schema compliance is computed and in [0, 1].
- Exits 0 on success.

---

## Phase 5 — Baseline Comparison and Ablations

**Objective:** Run the monolithic-VLM baseline and the ablations that justify each design decision, producing the load-bearing comparison.

**Files to Create / Modify:**
- `src/c3_explanation/model/generate_report.py` (extend with a `grounding_enabled` flag and a monolithic baseline mode)
- `src/c3_explanation/retrieval/__init__.py`
- `src/c3_explanation/retrieval/action_retriever.py`
- `src/c3_explanation/metrics/report.py` (extend to write `ablations.csv`)
- `src/c3_explanation/run_c3_ablations.py`
- `tests/verify_c3_phase_5.py`

**Detailed Instructions & Logic:**
1. Baseline mode: `generate` with `grounding_enabled=false` and no fine-tuned adapter (zero-shot base VLM, image + generic prompt only) — the monolithic baseline.
2. `action_retriever.py`: optional retrieval grounding for `recommended_action` (RAFT-style), gated behind a config flag; a simple indexed store of reference actions keyed by defect type.
3. `run_c3_ablations.py` drives, each toggled by a flag so it can be isolated:
   - **Grounding on/off** (flagship — expect the largest hallucination change).
   - Fine-tuned vs zero-shot base.
   - Structured JSON vs free-form output.
   - Retrieval on/off for the recommendation field.
   - Corpus enrichment on/off.
   - (Optional) 7B vs 3B.
4. Write `outputs/tables/c3/ablations.csv` with one row per ablation cell across all four metric axes.

**Integration requirements:** Reuses Phase 4 metrics. Baseline and full system see identical images and grounding inputs for fairness. Logs to MLflow.

**Edge Case & Windows Protections:**
- The baseline must receive exactly the same test images as the full system.
- `num_workers=0` throughout.
- Deterministic seeding so ablation deltas are attributable to the toggled factor, not noise.

**Verification Script & Acceptance Test:** `tests/verify_c3_phase_5.py`
- Runs the grounding-on vs grounding-off ablation on a small slice.
- Asserts `ablations.csv` gains rows with finite metrics for both settings across all four axes.
- Asserts the baseline and full system were evaluated on the identical image set (assert equal path sets).
- Exits 0 on success. (Full ablation sweep run manually after the test passes.)

---

## Phase 6 — End-to-End Demonstration, Figures, and Reproducibility Lock

**Objective:** Show the whole pipeline working as one (C2 → C1 → C3), produce publication-quality figures, enforce the Windows/`num_workers` invariant, and lock the component.

**Files to Create / Modify:**
- `src/c3_explanation/pipeline/__init__.py`
- `src/c3_explanation/pipeline/end_to_end.py`
- `src/c3_explanation/metrics/report.py` (final figure/summary helpers)
- `docs/C3_RESULTS.md`
- `tests/verify_c3_phase_6.py`

**Detailed Instructions & Logic:**
1. `end_to_end.py`: take a sample of C2 synthetic defects, run C1 to detect/localise, run the C3 bridge + fine-tuned model to produce reports — the full integrative demonstration. This is qualitative (a montage + example reports), not a scored metric.
2. Final figures (300 dpi) to `outputs/figures/c3/`: example reports alongside images and masks; the decoupled-vs-monolithic bar chart across the four metric axes; the grounding-on/off hallucination chart; the end-to-end C2→C1→C3 montage.
3. `docs/C3_RESULTS.md`: a short auto-generated summary pointing to the metric CSVs and key figures, with headline numbers filled in.

**Integration requirements:** Reuses all prior phases; reads C2 synthetic outputs and C1 read-only.

**Edge Case & Windows Protections:**
- Repo-wide guard: grep `src/c3_explanation/` for `num_workers=` and assert every occurrence is `0`. Fail the test otherwise.
- Confirm every entry point loads config from YAML (no hardcoded hyperparameters).
- Confirm generation uses deterministic decoding.

**Verification Script & Acceptance Test:** `tests/verify_c3_phase_6.py`
- Runs the end-to-end pipeline on a small sample of C2 synthetic defects and asserts a schema-valid report is produced for each.
- Runs the repo-wide `num_workers=0` grep assertion over `src/c3_explanation/`.
- Asserts the expected figure files exist for at least one dataset.
- Asserts `docs/C3_RESULTS.md` exists and references the metric CSVs.
- Exits 0 on success. Tag `c3-complete` after it passes.

---

## Definition of Done (whole component)
- [ ] C3 phase tests `verify_c3_phase_0` … `verify_c3_phase_6` all exit 0.
- [ ] Decoupled pipeline: C1 grounding → Qwen2.5-VL QLoRA → schema-valid JSON reports.
- [ ] Constructed, versioned report corpus with C2 held out.
- [ ] Four-axis evaluation on held-out real defects: schema compliance, factual accuracy, hallucination rate, rubric quality.
- [ ] Decoupled-vs-monolithic baseline comparison, with grounding on/off as the flagship ablation.
- [ ] End-to-end C2 → C1 → C3 demonstration figure.
- [ ] Everything seeded, config-driven, MLflow-logged, DVC-tracked, `num_workers=0` throughout.
- [ ] `docs/C3_RESULTS.md` summarises the headline outcome and links artifacts.
- [ ] Git tag `c3-complete`.

---

## First Prompt to Codex (orientation, no code)

Use this as the opening message, exactly as the C2 build began:

> You are building Component 3 (C3: Vision-Language Defect Explanation) into an existing MSc dissertation codebase. Before writing any code:
> 1. Read `docs/C3_SYSTEM_SPEC.md` in full — the permanent architecture anchor.
> 2. Read `docs/C3_EXECUTION_PLAN.md` in full — the phase-by-phase build sequence (Phases 0–6). You implement exactly one phase at a time, in order.
> 3. Inspect the existing repository without modifying anything. Confirm the presence and shape of: `src/c1_detector/`, `src/c2_synthesis/`, `data/processed/{mvtec_ad,ECF-Dataset}/`, `data/splits/{mvtec_ad,ecf}_splits.json`, and the existing `outputs/logs/mlflow` store. Report what you find so we can confirm the C1/C2 integration points.
> Then summarise in your own words: (a) what C3 does and why it is decoupled, (b) the non-negotiable engineering constraints from the spec, and (c) exactly what Phase 0 requires and how its acceptance test verifies success — with particular attention to the `bitsandbytes` 4-bit risk on Windows and the fallback chain.
> Do NOT write any code yet. Do NOT modify any file under `src/c1_detector/`, `src/c2_synthesis/`, or `data/`. Wait for my confirmation before starting Phase 0.
