# C2 Execution Plan
## Phase-by-Phase Build Prompts for Codex

**Companion to:** `docs/C2_SYSTEM_SPEC.md` (the architecture anchor — read it first).
**Audience:** OpenAI Codex, building Component 2 incrementally into the existing repository.
**Golden rules for every phase:**
- Read `docs/C2_SYSTEM_SPEC.md` before writing code. It is the single source of truth for equations, paths, and contracts.
- Do **not** modify any C1 source (`src/c1_detector/`, `src/data/`) or any file under `data/`. C2 consumes them read-only.
- **Every** `DataLoader` uses `num_workers=0` (Windows spawn deadlock protection).
- Precision is **bf16**; VAE decode is guarded against NaN/black frames.
- No magic numbers — all hyperparameters come from YAML configs.
- A phase is complete only when its `tests/verify_phase_X.py` exits 0. Do not proceed otherwise.
- After each artifact-producing phase, stage DVC/git as specified in the spec.

---

## Phase 0 — Environment, Scaffolding, and Base-Model Sanity

**Objective:** Stand up the C2 package skeleton, install and pin the generative stack, and prove the SD 1.5 inpainting pipeline loads and performs one clean inpaint round-trip in bf16 without a black/NaN decode.

**Files to Create / Modify:**
- `src/c2_synthesis/__init__.py`
- `src/c2_synthesis/utils/__init__.py`
- `src/c2_synthesis/utils/seed.py`
- `src/c2_synthesis/utils/logging_utils.py`
- `src/c2_synthesis/utils/image_io.py`
- `src/c2_synthesis/utils/mlflow_utils.py`
- `src/c2_synthesis/configs/sd15_inpaint_base.yaml`
- `requirements_c2.txt` (additive; do not touch existing requirements)
- `tests/verify_phase_0.py`

**Detailed Instructions & Logic:**
1. Create the full `src/c2_synthesis/` package tree from the spec (all `__init__.py` files, empty sub-packages allowed for now).
2. `requirements_c2.txt` pins: `diffusers`, `transformers`, `accelerate`, `peft`, `safetensors`, `torchmetrics`, `lpips`, `clean-fid`, `pyyaml`, `mlflow` (match versions already in the env where possible; do not downgrade torch).
3. `utils/seed.py`: `set_global_seed(seed:int)` seeding `random`, `numpy`, `torch`, `torch.cuda`, and `PYTHONHASHSEED`.
4. `utils/logging_utils.py`: a `get_logger(name)` returning a console+file logger writing to `outputs/logs/c2/`.
5. `utils/image_io.py`: `load_image_rgb(path)`, `load_mask_binary(path)` (returns uint8 {0,1}, binarised at >127), `save_image`, `save_mask`. All use PIL, forward-slash POSIX paths.
6. `utils/mlflow_utils.py`: `start_c2_run(experiment, run_name, params)` context manager wrapping the existing MLflow store at `file:./outputs/logs/mlflow`.
7. `configs/sd15_inpaint_base.yaml`: base model id `runwayml/stable-diffusion-inpainting`, resolution 512, scheduler `DDIM`, `num_inference_steps` 50, `guidance_scale` 7.5, `dtype` `bfloat16`, `vae_decode_dtype` `float32`.

**Integration requirements:** MLflow store path must match C1's (`file:./outputs/logs/mlflow`). Do not create a second store.

**Edge Case & Windows Protections:**
- Load the pipeline with `torch_dtype=torch.bfloat16`, move to `cuda`.
- After the test inpaint, assert the decoded image contains no NaNs and is not all-black (mean pixel > small epsilon). If it fails, the spec mandates decoding the VAE in fp32; implement `vae_decode_dtype` handling so the config switch works.
- No DataLoaders yet; if any appear, `num_workers=0`.

**Verification Script & Acceptance Test:** `tests/verify_phase_0.py`
- Imports all new util modules successfully.
- Loads the SD 1.5 inpainting pipeline in bf16 onto CUDA.
- Runs one inpaint on a synthetic 512×512 grey image with a centred square mask and a trivial prompt.
- Asserts: output is 512×512 RGB, contains no NaN, and is not all-black. As a gross-failure check on the raw pipeline output, compute mean absolute difference over the unmasked `(1 - mask)` region with pixels normalised to `[0,1]` and assert it is below `0.1`. This is not the bit-identical pixel-composite guarantee, which is added only in Phase 3.
- Prints GPU name, peak VRAM, and elapsed time.
- Exits 0 on success.

---

## Phase 1 — Data Pairing and Patch Extraction (C1 Integration)

**Objective:** Turn C1's processed data and split manifests into C2's inputs: per-class few-shot (defect image, mask) pairs at budgets {5,10,20,all}, a clean-image target pool, and 256×256 micro-defect crops for ECF.

**Files to Create / Modify:**
- `src/c2_synthesis/configs/datasets/mvtec.yaml`
- `src/c2_synthesis/configs/datasets/ecf.yaml`
- `src/c2_synthesis/data/__init__.py`
- `src/c2_synthesis/data/pair_builder.py`
- `src/c2_synthesis/data/patch_extractor.py`
- `tests/verify_phase_1.py`

**Detailed Instructions & Logic:**
1. Dataset YAMLs declare: logical dataset key, processed-data path, manifest path, class list, excluded classes (ECF: `pseudo_broken_solder`, `Serious defect`, and `solder_bead`), generation mode (`whole` for MVTec, `patch` for ECF), and crop size (256 for ECF). Physical dataset paths are read only from these YAMLs.
2. `pair_builder.py`:
   - `build_pairs(dataset, category, budget)` reads the C1 split manifest (read-only), returns lists of `(image_path, mask_path)` for real defects and a separate list of clean `train/good` image paths.
   - Budget selection is **deterministic** (seeded shuffle then take first N) so {5,10,20} are nested subsets of `all`.
   - Validates every pair: image exists, mask exists, mask is non-empty (contains defect pixels), dimensions match.
3. `patch_extractor.py`:
   - `extract_defect_crop(image, mask, size=256)` returns a crop centred on the mask centroid, with edge clamping, plus `crop_offset` and `crop_bbox` for later compositing (per spec §3.4).
   - `composite_crop_back(full_clean, gen_crop, crop_offset, crop_mask, crop_bbox)` resizes an oversized crop back to `crop_bbox` when required, then places it into a full clean frame using the pixel-composite equation.

**Integration requirements:** Read `data/splits/*_splits.json` exactly as C1 wrote them. Never modify them. Never re-derive splits. Use `utils/image_io` for all I/O.

**Edge Case & Windows Protections:**
- Empty or all-zero masks → skip with a logged warning, do not crash.
- ECF defects where a bbox exceeds 256 → use the tightest containing square and resize when that square fits in-frame. If it cannot fit, drop the sample with a logged warning and increment the class's crop-exclusion count.
- POSIX paths throughout for cross-platform manifest portability.

**Verification Script & Acceptance Test:** `tests/verify_phase_1.py`
- Builds pairs for one MVTec class (e.g. `bottle`) and one ECF class (e.g. `1.scratch`) at all four budgets.
- Asserts budget nesting: the 5-set ⊂ 10-set ⊂ 20-set ⊂ all-set.
- Asserts every returned pair passes validation (files exist, mask non-empty, dims match).
- For ECF, extracts one crop and composites it back into a clean frame; asserts the composite equals the clean frame outside the crop region (bit-identical).
- Saves a visual contact sheet (`outputs/figures/c2/phase1_pairs_{class}.png`) showing defect images with mask overlays for human eyeballing.
- Exits 0 on success.

---

## Phase 2 — LoRA + Token Fine-Tuning on One Pilot Class

**Objective:** Implement the three-term DefectFill loss, the learned-token manager, and the LoRA training loop; validate the whole training path end-to-end on **one** easy pilot class before scaling.

**Files to Create / Modify:**
- `src/c2_synthesis/configs/lora_defect.yaml`
- `src/c2_synthesis/train/__init__.py`
- `src/c2_synthesis/train/losses.py`
- `src/c2_synthesis/train/token_manager.py`
- `src/c2_synthesis/train/train_lora_defect.py`
- `tests/verify_phase_2.py`

**Detailed Instructions & Logic:**
1. `lora_defect.yaml`: rank 16, alpha 16, target modules = UNet cross-attention (`to_q,to_k,to_v,to_out.0`), lr 1e-4, max_steps 1000 (pilot), loss weights `lambda_obj: 1.0`, `lambda_attn: 0.1`, `alpha: 0.1`, batch size 1, gradient_accumulation 1.
2. `losses.py`: implement exactly the spec §4 equations.
   - `defect_loss(noise, noise_pred, mask_latent)` → masked MSE (§4.1).
   - `object_loss(noise, noise_pred, mask_latent, alpha)` with `M' = M + alpha*(1-M)` (§4.2).
   - `attention_loss(attn_map_token, mask_latent)` (§4.3).
   - `total_c2_loss(...)` combining with config weights (§4.4).
   - Every function documented with the equation it implements.
3. `token_manager.py`: add a new token (e.g. `<mvtec-bottle-defect>`) to the CLIP tokenizer/text-encoder embedding table; expose its embedding as trainable; save/load to `token.pt`.
4. `train_lora_defect.py`:
   - Loads base pipeline (bf16), freezes VAE/text-encoder (except the new token embedding), injects LoRA via `peft` on the UNet.
   - Captures the token's cross-attention map via forward hooks for the attention loss.
   - Trains with the three-term loss; logs each term separately to MLflow.
   - Saves `lora.safetensors` + `token.pt` to `outputs/checkpoints/c2/{dataset}/{class}/`.
   - `num_workers=0`; bf16; seeded.

**Integration requirements:** Consumes Phase 1 pairs. Uses `mlflow_utils` and `seed`. Writes to the spec's checkpoint path.

**Edge Case & Windows Protections:**
- Before the full pilot run, execute a **timed 50-step smoke test** on five MVTec `bottle` pairs and print steps/sec + projected 1000-step time (C1 Dinomaly lesson). Abort with a clear message if the projection exceeds the config-exposed `max_projected_pilot_minutes` threshold (default: 120 minutes).
- Guard the attention-map hook against shape mismatches (log and skip attention loss for that step rather than crash).
- bf16 forward/backward; keep loss computation in fp32 for numerical stability.

**Verification Script & Acceptance Test:** `tests/verify_phase_2.py`
- Runs a short (e.g. 50-step) training on the pilot class (MVTec `bottle` or `hazelnut`).
- Asserts all three loss terms are finite and the total loss decreases over the run (compare mean of first 10 vs last 10 steps).
- Asserts `lora.safetensors` and `token.pt` are written and reloadable.
- Prints steps/sec and projected 1000-step wall time.
- Exits 0 on success.

---

## Phase 3 — Generation and Low-Fidelity Selection

**Objective:** Generate mask-aligned synthetic defects from a trained LoRA+token using latent + pixel compositing, then filter weak samples with the LPIPS-based LFS gate. Produce the (image, mask, meta) triples the downstream stage consumes.

**Files to Create / Modify:**
- `src/c2_synthesis/generate/__init__.py`
- `src/c2_synthesis/generate/generate_defects.py`
- `src/c2_synthesis/generate/low_fidelity_selection.py`
- `src/c2_synthesis/data/mask_bank.py`
- `tests/verify_phase_3.py`

**Detailed Instructions & Logic:**
1. `mask_bank.py`:
   - `real_mask_transplant(clean_image, real_mask, location)` — place a real defect mask onto a clean image at a plausible, in-bounds location (primary strategy, spec §3.2a).
   - Stub `generate_masks(...)` for the optional mask-generator ablation (spec §3.2b) — implement interface now, full model later.
2. `generate_defects.py`:
   - Loads base pipeline + a class's LoRA + token.
   - For each (clean image, transplanted mask): runs inpainting with **latent-space compositing** each denoising step (spec §4.5), decodes with the VAE decode safeguard, then applies **pixel-space compositing** (spec §4.6) so all non-defect pixels are bit-identical to the clean source.
   - ECF: operates on 256 crops then composites back to full frame via `patch_extractor.composite_crop_back`.
   - Writes (image, mask, meta) triples per the spec §3.5 contract, with full provenance and seed.
3. `low_fidelity_selection.py`:
   - `lfs_score(gen_image, clean_image, mask)` = masked LPIPS (spec §4.7).
   - `filter_batch(samples, percentile=25)` rejects samples below the per-class percentile threshold; logs acceptance rate.

**Integration requirements:** Uses Phase 2 checkpoints and Phase 1 crop/composite utilities. Writes to `outputs/synthetic/{dataset}/{class}/`.

**Edge Case & Windows Protections:**
- VAE decode safeguard active (fp32 fallback on NaN).
- Pixel composite must guarantee `(1-m)` region equals the clean source exactly — assert this in the test.
- Any `DataLoader` batching: `num_workers=0`.
- Seed every sample; record in meta.

**Verification Script & Acceptance Test:** `tests/verify_phase_3.py`
- Generates ~30 samples for the pilot class using the Phase 2 checkpoint.
- Asserts each output triple exists and meta validates against the §3.5 schema.
- Asserts the pixel-composite invariant: for a sample, `(1-mask)*generated == (1-mask)*clean` exactly.
- Runs LFS; asserts acceptance rate is logged and 0 < rate ≤ 1.
- Saves a samples grid (`outputs/figures/c2/phase3_samples_{class}.png`).
- Exits 0 on success.

---

## Phase 4 — Scale Generation to All Classes + Fidelity Metrics

**Objective:** Train LoRAs and generate filtered synthetic sets for every retained class across MVTec and ECF, then compute per-class generative fidelity (FID, KID, LPIPS diversity) and emit the fidelity table.

**Files to Create / Modify:**
- `src/c2_synthesis/metrics/__init__.py`
- `src/c2_synthesis/metrics/fid_kid.py`
- `src/c2_synthesis/metrics/lpips_diversity.py`
- `src/c2_synthesis/metrics/report.py`
- `src/c2_synthesis/train/sweep_train_all.py`
- `src/c2_synthesis/generate/sweep_generate_all.py`
- `tests/verify_phase_4.py`

**Detailed Instructions & Logic:**
1. `sweep_train_all.py`: iterate classes from the dataset YAMLs, skip excluded, train each LoRA+token (reusing Phase 2), write checkpoints. Timed smoke test before the first full run. Log per-class wall time.
2. `sweep_generate_all.py`: iterate classes, generate + LFS-filter a target count (e.g. 200 accepted) per class, write triples.
3. `fid_kid.py`: wrap `clean-fid`/`torchmetrics` for FID and KID between real defects and synthetic defects per class. **KID is the primary fidelity metric** (small-sample robustness, spec §5.1).
4. `lpips_diversity.py`: mean pairwise LPIPS among a class's synthetic samples (mode-collapse guard).
5. `report.py`: assemble `outputs/tables/c2/fidelity.csv` with columns `dataset,class,n_real,n_synth,crop_exclusion_count,FID,KID,LPIPS_diversity,lfs_acceptance`.

**Integration requirements:** Reuses Phases 2–3. All metrics logged to MLflow and written to the git-tracked CSV.

**Edge Case & Windows Protections:**
- Classes with very few real defects (ECF `missing_plate`, `metal_foreign_body`) → still run, flag low-n in the CSV, interpret KID cautiously.
- Report each ECF class's Phase 1 crop-exclusion count so the dropped out-of-frame-square limitation remains visible.
- `num_workers=0` for any batching in metric computation.
- DVC-add checkpoints + synthetic after the sweep.

**Verification Script & Acceptance Test:** `tests/verify_phase_4.py`
- Runs the sweep on a **small subset** (2 MVTec + 2 ECF classes) to keep the test fast.
- Asserts checkpoints and synthetic triples exist for each.
- Asserts `fidelity.csv` is written with finite FID/KID for each and no NaNs in required columns.
- Exits 0 on success. (Full sweep is run manually after the test passes.)

---

## Phase 5 — Downstream Utility: U-Net, 4 Configs, NPI Ramp Curve

**Objective:** The core result. Train a supervised U-Net (ResNet-34) under four data configurations at real-defect budgets {5,10,20,all}, evaluate on a fixed held-out **real** test set, and produce the downstream table + NPI ramp curve, foregrounding pixel-F1 for the small-defect analysis.

**Files to Create / Modify:**
- `src/c2_synthesis/downstream/__init__.py`
- `src/c2_synthesis/downstream/unet_segmenter.py`
- `src/c2_synthesis/downstream/datamodule.py`
- `src/c2_synthesis/downstream/train_downstream.py`
- `src/c2_synthesis/downstream/evaluate.py`
- `src/c2_synthesis/downstream/plot_ramp_curve.py`
- `tests/verify_phase_5.py`

**Detailed Instructions & Logic:**
1. `unet_segmenter.py`: U-Net with a ResNet-34 encoder (segmentation-models-pytorch or a clean local impl). Binary defect segmentation head.
2. `datamodule.py`: builds a training set for a given `(config, budget)`:
   - `real-only`: the budget's real defects only.
   - `synthetic-only`: C2 synthetic only.
   - `real+synthetic`: union.
   - `real+classical-aug`: real + rotations/flips/CutPaste-style paste (implement a simple classical augmenter).
   - The **test set is always the fixed held-out real defects** from the C1 manifest, never seen in training. `num_workers=0`.
3. `train_downstream.py`: trains a U-Net for one `(config, budget, seed)`; identical hyperparameters across configs — only the data changes. ≥3 seeds. Logs to MLflow.
4. `evaluate.py`: on the real test set computes image-level AUROC & AP and pixel-level AUROC, AUPRO, and **pixel-F1 (max)**. Writes `outputs/tables/c2/downstream.csv` with `dataset,class,config,budget,seed,I_AUROC,AP,P_AUROC,AUPRO,pixel_F1`.
5. `plot_ramp_curve.py`: produces the NPI ramp curve figure (x = budget, y = metric, one line per config) at 300 dpi to `outputs/figures/c2/npi_ramp_{dataset}_{metric}.png`. Also produces the small-defect pixel-F1 comparison figure for ECF micro-defect classes (spec §5.4 / C1 linkage).

**Integration requirements:** Reads real defects and clean/test images from C1 manifests; reads synthetic from Phase 3/4 outputs. Uses `seed`, `mlflow_utils`.

**Edge Case & Windows Protections:**
- `num_workers=0` on all three dataloaders — asserted in the test.
- Timed smoke test (1 epoch) before the full sweep; print projected total time.
- Fixed test set integrity: assert zero overlap between any training image path and the test set.
- ≥3 seeds; report mean ± std (single runs are not trustworthy).

**Verification Script & Acceptance Test:** `tests/verify_phase_5.py`
- Runs a minimal matrix: one class, budgets {5, all}, all four configs, 1 seed, few epochs.
- Asserts train/test disjointness (no shared image paths).
- Asserts `downstream.csv` populated with finite metrics for every (config, budget) cell.
- Asserts the ramp-curve figure file is created.
- Asserts every DataLoader used `num_workers=0` (introspect or assert via a shared factory).
- Exits 0 on success.

---

## Phase 6 — Ablations, Final Figures, and Reproducibility Lock

**Objective:** Run the ablations that justify each design decision, generate all publication-quality figures, enforce the Windows/`num_workers` invariant repo-wide, and lock the component for reproducibility.

**Files to Create / Modify:**
- `src/c2_synthesis/ablations/__init__.py`
- `src/c2_synthesis/ablations/run_ablations.py`
- `src/c2_synthesis/metrics/report.py` (extend to write `ablations.csv`)
- `docs/C2_RESULTS.md` (auto-summarised results anchor)
- `tests/verify_phase_6.py`

**Detailed Instructions & Logic:**
1. `run_ablations.py` drives (each isolating one spec decision):
   - **Loss ablation:** drop `L_obj`, then `L_attn`; measure KID + downstream pixel-F1 delta.
   - **LoRA rank sweep:** {8, 16, 32}.
   - **Mask source:** real-transplant vs generated masks.
   - **ControlNet on/off** on one structured ECF class (optional if time).
   - **LFS on/off:** quantify the quality-gate value.
   - **(Optional) SD 1.5 vs SDXL** on one class.
2. Write `outputs/tables/c2/ablations.csv` with one row per ablation cell.
3. Generate final figures: samples grids, NPI ramp curves, ablation bar charts, small-defect pixel-F1 comparison — all 300 dpi to `outputs/figures/c2/`.
4. `docs/C2_RESULTS.md`: a short auto-generated summary pointing to the three CSVs and the key figures, with the headline numbers filled in.

**Integration requirements:** Reuses all prior phases. All results to MLflow + git-tracked CSVs. DVC-add any new checkpoints/synthetic.

**Edge Case & Windows Protections:**
- Repo-wide guard: grep `src/c2_synthesis/` for `num_workers=` and assert every occurrence is `0`. Fail the test otherwise.
- Re-assert VAE decode safeguard and bf16 usage are present in generation code.
- Confirm every entry point loads config from YAML (no hardcoded hyperparameters) — a lightweight static check.

**Verification Script & Acceptance Test:** `tests/verify_phase_6.py`
- Runs one ablation cell end-to-end (e.g. LFS on vs off on the pilot class) and asserts `ablations.csv` gains rows with finite metrics.
- Runs the repo-wide `num_workers=0` grep assertion over `src/c2_synthesis/`.
- Asserts all expected figure files exist for at least the pilot dataset.
- Asserts `docs/C2_RESULTS.md` exists and references the three CSVs.
- Exits 0 on success. Tag `c2-complete` after it passes.

---

## Definition of Done (whole component)
- [ ] Phases 0–6 verification scripts all exit 0.
- [ ] Mask-aligned, LFS-filtered synthetic (image, mask, meta) sets for MVTec + ECF, DVC-tracked.
- [ ] `fidelity.csv`, `downstream.csv`, `ablations.csv` populated and git-tracked.
- [ ] NPI ramp curve + small-defect pixel-F1 figures at 300 dpi.
- [ ] Everything seeded, config-driven, MLflow-logged, DVC-versioned, `num_workers=0` throughout.
- [ ] `docs/C2_RESULTS.md` summarises the headline outcome and links artifacts.
- [ ] Git tag `c2-complete`.
