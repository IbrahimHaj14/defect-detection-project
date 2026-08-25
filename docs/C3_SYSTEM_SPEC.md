# C3 System Specification
## Component 3: Vision-Language Defect Explanation — Architectural Blueprint

**Project:** Vision-Language Defect Detection for New Product Introduction (NPI)
**Institution:** MSc Applied AI, Warwick WMG
**Status:** Permanent architecture anchor. This document defines *what* C3 is. The companion `C3_EXECUTION_PLAN.md` defines *how* it is built, phase by phase.
**Hardware target:** NVIDIA RTX PRO 5000 Blackwell, 48 GB VRAM, native bf16 · Windows 11 · conda env `defect-detect`

---

## 1. System Purpose and Scope

### 1.1 The problem C3 solves
C1 produces an anomaly map and a defect region; C2 produces synthetic training data. Neither produces anything a human on the factory floor can act on directly. A raw heatmap is not a decision. C3 converts the detector's numerical output into a structured, human-readable defect report stating what the defect is, where it is, how severe it is, and what action is recommended.

At New Product Introduction this is the component that makes the pipeline usable by the people running the line, who are often not the people who built the inspection system and who face defect types that are new to everyone. C3 turns an opaque signal into an auditable explanation.

### 1.2 The core commitment
C3 is a **human-in-the-loop aid, not an autonomous decision-maker.** This is an evidence-based scope decision: current vision-language models fall short of the reliability that unattended industrial decisions demand (on the MMAD benchmark even GPT-4o reaches only 74.9% average accuracy). C3 produces a report a human reads, verifies against the image, and overrides when wrong, and it is evaluated on whether that report is faithful and useful — not on whether it could replace the human.

### 1.3 Scope: an evaluated proof-of-concept
The deliverable is a working, evaluated proof-of-concept: a decoupled pipeline producing schema-valid, grounded reports on real C1 outputs that measurably beats a monolithic-VLM baseline on hallucination and report quality. It is explicitly **not** a claim of a deployable, fully-reliable industrial reporting system. This scope is a deliberate strength, stated plainly wherever C3 is described.

### 1.4 Method in one paragraph
C3 is a **decoupled "detect-then-describe" pipeline**. C1 provides the visual evidence (anomaly map, defect region). A deterministic, non-learned **grounding bridge** converts that evidence into structured text facts (bounding box, centroid, area fraction, plain-language location, severity by rule). A **vision-language model (Qwen2.5-VL-7B, adapted with QLoRA)** receives the image plus those grounded facts and emits a **schema-constrained JSON report**. The language model is never asked to independently find the defect; it turns verified evidence into words. Hallucination is suppressed by the grounding and the fixed schema, and measured rather than assumed.

---

## 2. Architecture Specification

### 2.1 The three internal stages (Eyes / Bridge / Brain)

```
  C1 output                Grounding bridge              Vision-language model
 (anomaly map,   ──────►   (deterministic,      ──────►  (Qwen2.5-VL-7B + QLoRA)
  binary mask)             no parameters)                 image + facts -> JSON report
   [EYES]                    [BRIDGE]                        [BRAIN]
```

1. **Eyes — evidence (exists already).** C1 produces a binary defect mask and anomaly map for an image. C3 consumes these read-only; it does not re-run detection.
2. **Bridge — grounding (deterministic, no parameters).** Converts the mask into structured facts: bounding box, centroid, area as a fraction of the image, a plain-language region label, and a rule-assigned severity. This stage has no learned parameters and therefore cannot hallucinate — it is arithmetic on the mask. It is the single largest hallucination reducer in the system and is treated as the most important stage.
3. **Brain — description (learned).** Qwen2.5-VL-7B, adapted with QLoRA, receives the image and the grounded facts and generates the JSON report. Its factual fields are dictated by the bridge; its contribution is the description and recommendation language.

### 2.2 Why decoupled rather than monolithic
A single VLM asked to both find and describe defects hallucinates heavily on industrial images, because its vision encoder down-samples the input and suppresses small, low-contrast defects, and asked to find what it cannot see it invents. The decoupled design hands the model the spatial truth as text so the report's facts are as reliable as C1, confining the VLM's fallibility to phrasing. A monolithic VLM is retained only as the **baseline to beat** in evaluation.

### 2.3 Model specification
- **Base model:** `Qwen/Qwen2.5-VL-7B-Instruct` (open-weight; native dynamic-resolution vision; direct precedent for defect description). Documented fallback: the 3B variant if the 4-bit backend is unavailable on Windows or VRAM/time demand it.
- **Adaptation:** QLoRA — 4-bit NF4 quantised base, low-rank adapters trained in bf16. Adapters target the language-model attention projections; the vision tower stays frozen and quantised.
- **Quantisation:** 4-bit NF4 via `bitsandbytes`. If unavailable on Windows, fall back to 8-bit, then to bf16 3B (the 48 GB card allows any of these).

### 2.4 Component inventory

| Component | Trained? | Role |
|---|---|---|
| C1 detector output | frozen (external) | Supplies mask + anomaly map (Eyes) |
| Grounding bridge | no parameters | Mask → structured text facts (Bridge) |
| Severity rule engine | no parameters | Measured area + type → severity level |
| Qwen2.5-VL vision tower | frozen, 4-bit | Encodes the image |
| Qwen2.5-VL language model | frozen base + QLoRA adapters | Generates the report (Brain) |
| Retrieval index (optional) | no parameters | Grounds recommendation field (ablation) |
| Rubric / LLM-as-judge | external | Evaluation only, never in the pipeline |

### 2.5 Repository layout

C3 is additive. It must not modify C1 or C2 source and consumes their outputs read-only.

```
defect-detection-project/
├── docs/
│   ├── C3_SYSTEM_SPEC.md              # this document
│   └── C3_EXECUTION_PLAN.md           # phase-by-phase Codex prompts
├── src/
│   ├── c1_detector/                   # EXISTING, read-only for C3
│   ├── c2_synthesis/                  # EXISTING, read-only for C3
│   └── c3_explanation/                # NEW — all C3 code
│       ├── __init__.py
│       ├── configs/
│       │   ├── qwen_vl_base.yaml       # model id, quantisation, dtype, gen params
│       │   ├── qlora_report.yaml       # rank, alpha, lr, steps, target modules
│       │   ├── report_schema.yaml      # the fixed JSON schema + severity thresholds
│       │   └── datasets/
│       │       ├── mvtec.yaml          # class list, defect-type -> template map
│       │       └── ecf.yaml            # dataset_key: ecf, processed_dir: ECF-Dataset
│       ├── grounding/
│       │   ├── __init__.py
│       │   ├── bridge.py               # C1 mask -> structured facts (deterministic)
│       │   └── severity_rules.py       # measured area + type -> severity level
│       ├── data/
│       │   ├── __init__.py
│       │   ├── report_builder.py       # templated ground-truth reports
│       │   ├── enrichment.py           # optional offline description rephrasing
│       │   └── corpus_split.py         # train/test split; C2 held out
│       ├── model/
│       │   ├── __init__.py
│       │   ├── load_vlm.py             # Qwen2.5-VL load + QLoRA setup
│       │   ├── train_report_qlora.py   # fine-tuning entry point
│       │   └── generate_report.py      # image + grounding -> JSON report
│       ├── retrieval/
│       │   ├── __init__.py
│       │   └── action_retriever.py     # optional RAFT grounding (ablation)
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── schema_compliance.py
│       │   ├── factual_accuracy.py
│       │   ├── hallucination.py
│       │   ├── report_quality.py       # rubric / LLM-as-judge
│       │   └── report.py               # metric tables -> CSV
│       ├── pipeline/
│       │   ├── __init__.py
│       │   └── end_to_end.py           # C2 -> C1 -> C3 demonstration
│       └── utils/
│           ├── __init__.py
│           ├── seed.py
│           ├── logging_utils.py
│           ├── json_io.py              # safe JSON load/save/validate
│           └── mlflow_utils.py
├── tests/
│   ├── verify_phase_0.py … verify_phase_6.py   # C3 phase gates (see execution plan)
└── outputs/
    ├── checkpoints/c3/{dataset}/           # QLoRA adapter + tokenizer, DVC-tracked
    ├── corpus/c3/{dataset}/{train,test}/   # constructed report corpus, DVC-tracked
    ├── reports/c3/{dataset}/{class}/*.json # generated reports, DVC-tracked
    ├── tables/c3/                          # metric CSVs, git-tracked
    ├── figures/c3/                         # git-tracked (300 dpi)
    └── logs/mlflow/                        # existing MLflow store
```

**Import rule:** C3 imports from `src.c3_explanation.*` and may import C1/C2 utilities read-only. C1 and C2 never import C3.

---

## 3. Data Schemas and Contracts

### 3.1 Inputs C3 reads (read-only)

**C1 masks and anomaly maps.** C3 reads the C1 ground-truth masks (for corpus construction) and C1 predicted masks/anomaly maps (for inference-time grounding). Mask format follows the project contract: single-channel binary PNG, 255 = defect, 0 = normal, stem matches the image stem, binarised defensively on load (`>127`).

**Split manifests.** C3 reads `data/splits/{mvtec_ad,ecf}_splits.json` exactly as C1 wrote them, to know which images are defective and where their masks are. It never re-derives splits.

**Dataset key vs directory (carried from C2).** The logical key is `ecf`; the on-disk processed directory is `data/processed/ECF-Dataset`; the manifest is `data/splits/ecf_splits.json`. This mapping lives only in `configs/datasets/ecf.yaml` (`dataset_key`, `processed_dir`, `splits_path`) and is never hardcoded elsewhere. MVTec parallels with `mvtec_ad` / `data/processed/mvtec_ad` / `data/splits/mvtec_ad_splits.json`.

### 3.2 The report schema (the central contract)

Defined once in `configs/report_schema.yaml` and used identically for corpus construction, generation, and evaluation:

```json
{
  "defect_present": true,
  "defect_type": "scratch",
  "location": {
    "region": "upper-left quadrant",
    "bounding_box": [x, y, w, h],
    "centroid": [cx, cy]
  },
  "severity": {
    "level": "none | minor | moderate | severe",
    "affected_area_pct": 2.4,
    "rationale": "short text grounded in the measured area and defect type"
  },
  "description": "one or two sentences on the defect's visual appearance",
  "recommended_action": {
    "action": "inspect | rework | scrap | monitor",
    "reason": "short generated text"
  },
  "confidence": "how confident the report is, and on what basis"
}
```

`recommended_action` is split into a strict categorical `action` and free-text `reason`. For an empty mask,
`defect_present` is `false`; `location.bounding_box`, `location.centroid`, and `location.region` are all `null`;
`severity.level` is `none`; and `severity.affected_area_pct` is `0.0`. The `none` severity value is valid only when
`defect_present` is `false`; `assign_severity()` continues to return only `minor | moderate | severe` for actual
defects.

The common report schema uses the union of the separately configured MVTec and ECF defect-type vocabularies because
the report contains no dataset field. Corpus construction and report generation additionally enforce the applicable
dataset-specific vocabulary. The Phase 1 skeleton uses `recommended_action.action = "inspect"` only as an internal
schema-valid placeholder, not as a Bridge inference; Phase 2 corpus construction and Phase 4 final generation must
replace/populate it.

**Field provenance (which stage owns which field):**

| Field | Owner | Source |
|---|---|---|
| `defect_present` | Bridge | mask non-empty |
| `defect_type` | Bridge (corpus) / Brain (inference) | class label (corpus); model prediction constrained to known types (inference) |
| `location.bounding_box` | Bridge | computed from mask |
| `location.centroid` | Bridge | computed from mask |
| `location.region` | Bridge | centroid → quadrant/edge label |
| `severity.affected_area_pct` | Bridge | mask area / image area |
| `severity.level` | Bridge | severity rule engine |
| `severity.rationale` | Brain | generated, must cite the measured area |
| `description` | Brain | generated |
| `recommended_action.action` | Brain (optionally retrieval-grounded) | constrained enum |
| `recommended_action.reason` | Brain (optionally retrieval-grounded) | generated |
| `confidence` | Brain | generated |

The factual fields (location, area, severity level) are **owned by the deterministic bridge** and must not be overwritten or contradicted by the model. The Brain owns the explicitly marked fields, including inference-time `defect_type`, the constrained action category, and the free-text language fields.

### 3.3 Corpus example contract
Each training/eval example is a triple written atomically:
```
outputs/corpus/c3/{dataset}/{split}/{uid}.image.png     # or a path reference to the source image
outputs/corpus/c3/{dataset}/{split}/{uid}.grounding.json # bridge output (structured facts)
outputs/corpus/c3/{dataset}/{split}/{uid}.report.json    # ground-truth report (schema-valid)
```
`grounding.json` schema: `{uid, dataset, class, image_path, mask_path, bounding_box, centroid, area_pct, region, severity_level}`.

### 3.4 Generated-report contract
At inference, each report is written to `outputs/reports/c3/{dataset}/{class}/{uid}.json`, schema-valid, with a sidecar `{uid}.meta.json` recording model, adapter path, seed, generation parameters, and whether grounding/retrieval were enabled.

---

## 4. Corpus Generation Methodology

### 4.1 The problem
There is no ready-made dataset of defect images paired with reports. The corpus must be constructed. This is the C3 analogue of C2's mask-as-label insight and is solved before any fine-tuning.

### 4.2 Templated ground-truth reports
Every real defect already has a defect-type label, a ground-truth mask, and therefore a computable location and area. A ground-truth report is constructed deterministically for each:
- `defect_type` ← the known class label.
- `location.*`, `severity.affected_area_pct` ← computed from the mask by the same bridge used at inference.
- `severity.level` ← the documented threshold rule (`severity_rules.py`).
- `description`, `recommended_action` ← per-defect-type templates with slot-filling, correct by construction and consistent across the corpus.

This yields a large, perfectly-grounded corpus at zero annotation cost.

### 4.3 Optional fluency enrichment
An optional offline step (`enrichment.py`) rephrases the templated `description` into more natural language while preserving every grounded fact. Run once at corpus-construction time, never at inference. It is an ablation variable, not a default, and it must not alter any factual field.

### 4.4 Split discipline
The corpus is split so evaluation is on defects/images unseen during fine-tuning. **C2 synthetic defects are excluded from the C3 training corpus entirely** and reserved for the end-to-end demonstration (Section 6 of the roadmap), so C3 evaluation is on real defects with real grounding.

### 4.5 Severity rule (auditable, not a model guess)
`severity_rules.py` maps measured `affected_area_pct` and defect type to `minor | moderate | severe` by documented thresholds defined in `report_schema.yaml`. Severity is therefore consistent and auditable, never a subjective VLM output.

---

## 5. Hardware and Environment Constraints

### 5.1 Quantisation and precision
- **QLoRA:** 4-bit NF4 base + bf16 adapters. A 7B model in 4-bit is roughly 5–6 GB of weights, leaving large headroom on 48 GB.
- **bf16** for adapter compute and inference, consistent with C1/C2. Native on Blackwell.
- **`bitsandbytes` Windows risk:** the 4-bit backend has historically been fragile on Windows. Phase 0 must verify it imports and quantises before anything depends on it. Documented fallback order: 4-bit → 8-bit → bf16 3B.

### 5.2 Windows stability (carried from C1/C2, mandatory)
- **`num_workers=0`** on every DataLoader — spawn-deadlock protection. A repo-wide grep in the final phase asserts no nonzero value exists in `src/c3_explanation/`.
- **Timed smoke test before any long fine-tune** — short step count first, project full wall time, abort if implausible (the C1 Dinomaly lesson).
- **Local model caching** — the base VLM and tokenizer are cached locally so runs do not re-download; the model id and cache behaviour live in `qwen_vl_base.yaml`.
- **Disable Windows sleep** during long fine-tunes.

### 5.3 Determinism
- Global seed via `utils/seed.py` across `random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`.
- Generation records seed and parameters per report.
- Greedy or fixed-temperature decoding for reproducible reports; where a temperature > 0 is used, report over multiple samples.

### 5.4 Expected footprint
- QLoRA fine-tune of 7B: comfortably within 48 GB, likely under half.
- Inference: a few GB; report generation is seconds per image.

---

## 6. Evaluation Framework

All metrics are computed on a held-out test set of **real** defects with **real** C1 grounding. The monolithic-VLM baseline sees identical images and inputs.

### 6.1 Schema compliance
Fraction of generated reports that are valid against `report_schema.yaml`: parseable JSON, all required fields present, correct types, values in allowed ranges. The first, hardest gate — an unparseable report is unusable regardless of content.

### 6.2 Factual accuracy against ground truth
- `defect_type` accuracy vs the label.
- Location agreement vs the ground-truth mask (IoU of bounding box and/or centroid distance).
- `affected_area_pct` absolute error vs the measured area.
High by construction when grounding is respected — which is the point of the decoupled design, and what the grounding-on/off ablation demonstrates.

### 6.3 Hallucination rate (headline safety metric)
Fraction of reports containing a claim unsupported by the evidence: a defect type the image does not show, a location contradicting the mask, an invented feature. Scored by automatic contradiction checks against grounded fields plus a rubric on the free-text fields. This is where the decoupled architecture is expected to win decisively against the monolithic baseline.

### 6.4 Report quality (rubric / LLM-as-judge)
A fixed rubric scores usefulness, clarity, and correctness, applied by a human rater on a sample and/or an LLM-as-judge for scale. Captures helpfulness that the automatic metrics miss.

### 6.5 Comparison and ablations
Load-bearing experiment: **decoupled C3 vs monolithic-VLM baseline** on all four axes, identical inputs. Ablations isolate each decision:
- **Grounding on/off** (the bridge) — flagship ablation, largest expected hallucination change.
- Fine-tuned vs zero-shot base.
- Structured JSON vs free-form output.
- Retrieval on/off (recommendation field).
- Corpus enrichment on/off.
- (Optional) 7B vs 3B.

### 6.6 MLflow schema
- Experiments: `c3-train-{dataset}`, `c3-eval-{dataset}`, `c3-ablation-{name}`.
- Params: model id, quantisation, rank, alpha, lr, steps, seed, grounding/retrieval flags.
- Metrics: schema compliance, defect-type accuracy, location IoU, area error, hallucination rate, rubric score, wall time, peak VRAM.

### 6.7 Storage matrix (consistent with C1/C2)

| Artifact | Location | git | DVC | MLflow |
|---|---|---|---|---|
| QLoRA adapter + tokenizer | `outputs/checkpoints/c3/` | no | yes | path param |
| Report corpus | `outputs/corpus/c3/` | no | yes | no |
| Generated reports | `outputs/reports/c3/` | no | yes | no |
| Metric CSVs | `outputs/tables/c3/` | yes | no | metrics |
| Figures (300 dpi) | `outputs/figures/c3/` | yes | no | no |
| Params, metrics | — | no | no | yes |

---

## 7. Acceptance Philosophy

Every phase ends with a `tests/verify_phase_X.py` that must exit 0 before the next phase begins, mirroring the C1/C2 discipline: verify at every step, never scale on an unverified assumption. The execution plan specifies each phase's exact acceptance test. The `bitsandbytes` verification in Phase 0 and the grounding-correctness test in Phase 1 are the two highest-risk gates and must pass cleanly before any fine-tuning is attempted.
