# C2 System Specification
## Component 2: Diffusion-Based Defect Synthesis — Architectural Blueprint

**Project:** Vision-Language Defect Detection for New Product Introduction (NPI)
**Institution:** MSc Applied AI, Warwick WMG
**Status:** Permanent architecture anchor. This document defines *what* C2 is. The companion `C2_EXECUTION_PLAN.md` defines *how* it is built, phase by phase.
**Hardware target:** NVIDIA RTX PRO 5000 Blackwell, 48 GB VRAM, native bf16 · Windows 11 · conda env `defect-detect`

---

## 1. System Overview and Purpose

### 1.1 The problem C2 solves
At New Product Introduction (NPI) a factory has abundant defect-free images but almost no defect examples. Component 1 (PatchCore/Dinomaly) provides day-zero, label-free anomaly detection from normal images alone. It cannot, however, train a *supervised* detector — that still needs labelled defects that do not yet exist.

C2 closes this gap. It takes the first handful of real defects (5–20 per class) and synthesises a large, varied, **mask-aligned** set of realistic defect images. These let a supervised detector reach useful accuracy long before enough real defects would accumulate naturally.

### 1.2 The core commitment
C2 is evaluated on **downstream utility**, not image beauty. The load-bearing claim is:

> Adding C2 synthetic defects to a training set measurably improves a downstream supervised detector — most valuably on the small, sparse defects that pixel-F1 exposes and that matter most at NPI.

Generative-fidelity metrics (FID, KID, LPIPS) are reported but are secondary evidence. The primary result is the downstream U-Net experiment (Section 1.4) and the NPI ramp curve.

### 1.3 Method in one paragraph
C2 uses the **inpainting** paradigm on **Stable Diffusion 1.5 Inpainting**. For each defect class it fine-tunes a **LoRA** (rank 16, alpha 16) on the UNet cross-attention layers plus a **learned textual token** (e.g. `<ecf-scratch>`), trained with the **DefectFill three-term loss** (defect + object + attention). At inference it inpaints a synthesised defect into a masked region of a clean image; the conditioning mask doubles as the ground-truth segmentation label. Micro-defects (ECF) are handled by **256×256 patch-based** generation and composited back. A **Low-Fidelity Selection (LFS)** gate filters weak samples via masked LPIPS.

### 1.4 Downstream evaluation strategy
A supervised **U-Net (ResNet-34 encoder)** is trained under four data configurations and evaluated on a fixed, held-out **real** test set:
1. `real-only` — baseline to beat.
2. `synthetic-only` — synthesis alone (expected insufficient; honest control).
3. `real+synthetic` — the hypothesis (expected best).
4. `real+classical-aug` — fair baseline (rotations/flips/CutPaste), proving C2 beats cheap augmentation.

Each configuration is run at real-defect budgets **{5, 10, 20, all}** to produce the **NPI ramp curve**. Metrics: image-level AUROC/AP; pixel-level AUROC, AUPRO, and **pixel-F1 (max)** foregrounded for the small-defect analysis that links C2 back to C1's metric-divergence finding.

---

## 2. Repository Architecture and Directory Layout

C2 is additive. It must not modify C1 source. It consumes C1's processed data and split manifests read-only.

```
defect-detection-project/
│
├── docs/
│   ├── C2_SYSTEM_SPEC.md            # this document
│   └── C2_EXECUTION_PLAN.md         # phase-by-phase Codex prompts
│
├── src/
│   ├── data/                        # EXISTING C1 data pipeline (read-only for C2)
│   ├── c1_detector/                 # EXISTING C1 (read-only for C2)
│   └── c2_synthesis/                # NEW — all C2 code
│       ├── __init__.py
│       ├── configs/
│       │   ├── sd15_inpaint_base.yaml
│       │   ├── lora_defect.yaml
│       │   └── datasets/
│       │       ├── mvtec.yaml
│       │       └── ecf.yaml
│       ├── data/
│       │   ├── __init__.py
│       │   ├── pair_builder.py       # (defect img, mask) pairs from C1 splits
│       │   ├── patch_extractor.py    # 256x256 micro-defect crops (ECF)
│       │   └── mask_bank.py          # real-mask transplant + optional generator
│       ├── train/
│       │   ├── __init__.py
│       │   ├── losses.py             # defect / object / attention losses
│       │   ├── token_manager.py      # learned per-class defect tokens
│       │   └── train_lora_defect.py  # main fine-tuning entry point
│       ├── generate/
│       │   ├── __init__.py
│       │   ├── generate_defects.py   # inpainting inference -> (img, mask, meta)
│       │   └── low_fidelity_selection.py  # LPIPS quality gate
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── fid_kid.py
│       │   ├── lpips_diversity.py
│       │   └── report.py             # per-class metric tables -> CSV
│       ├── downstream/
│       │   ├── __init__.py
│       │   ├── unet_segmenter.py     # ResNet-34 encoder U-Net
│       │   ├── datamodule.py         # mixes real+synthetic per config
│       │   ├── train_downstream.py   # 4-config training harness
│       │   └── evaluate.py           # I-AUROC, P-AUROC, AUPRO, F1-max
│       └── utils/
│           ├── __init__.py
│           ├── seed.py               # global determinism
│           ├── logging_utils.py      # console + file logging
│           ├── mlflow_utils.py       # experiment tracking helpers
│           └── image_io.py           # safe load/save, mask binarisation
│
├── tests/
│   ├── verify_phase_0.py
│   ├── verify_phase_1.py
│   ├── verify_phase_2.py
│   ├── verify_phase_3.py
│   ├── verify_phase_4.py
│   ├── verify_phase_5.py
│   └── verify_phase_6.py
│
├── data/
│   ├── processed/{mvtec_ad,ecf}/...  # EXISTING C1 output (read-only)
│   └── splits/{mvtec_ad,ecf}_splits.json  # EXISTING (read-only)
│
└── outputs/
    ├── checkpoints/c2/{dataset}/{class}/
    │   ├── lora.safetensors           # DVC-tracked, gitignored
    │   └── token.pt                   # DVC-tracked, gitignored
    ├── synthetic/{dataset}/{class}/
    │   ├── images/*.png               # DVC-tracked
    │   ├── masks/*.png                # DVC-tracked
    │   └── meta/*.json                # DVC-tracked
    ├── tables/c2/
    │   ├── fidelity.csv               # git-tracked
    │   ├── downstream.csv             # git-tracked
    │   └── ablations.csv              # git-tracked
    ├── figures/c2/                    # git-tracked (300 dpi)
    └── logs/mlflow/                   # MLflow store (existing)
```

**Import rule:** C2 modules import from `src.c2_synthesis.*` and may import C1 data utilities read-only. C1 must never import C2. No circular dependencies.

---

## 3. Data Contracts and C1 Integration

### 3.1 Inputs C2 reads (all read-only from C1)

**Split manifests** — the single source of truth for which images belong to train/val/test:
```
data/splits/mvtec_ad_splits.json
data/splits/ecf_splits.json
```
Each manifest maps `category -> {stats, splits:{train,val,test:[{image_path,label,is_anomalous,mask_path}]}}`. C2 reads these exactly as C1 wrote them. C2 never re-derives splits.

**Clean images (inpainting targets):** the `is_anomalous=false` entries under each category's `train` split (i.e. `train/good/`). Hundreds per category, giving near-unlimited background diversity.

**Real defect pairs (few-shot conditioning):** the `is_anomalous=true` entries under `test`, each carrying an `image_path` and a non-null `mask_path`.

### 3.2 Mask format contract
- Binary PNG, single channel, values in {0, 255}.
- 255 = defect pixel, 0 = normal pixel.
- Mask filename stem matches its image stem (established in C1).
- Mask spatial dimensions equal the source image dimensions before any resize.
- C2 binarises defensively on load: `mask = (pixel > 127) ? 1 : 0`.

### 3.3 Dataset-specific handling

| Property | MVTec AD | ECF |
|---|---|---|
| Defect scale | Large / macro | Small / micro |
| Generation mode | Whole-image (512×512) | Patch-based (256×256 crops) |
| Mask layout | Per-category `ground_truth/<defect>/` | Per-category (post C1 `step_02c` restructure) |
| Compositing | Direct | Crop → generate → composite back to full frame |
| Excluded classes | none | `pseudo_broken_solder` (0 anomalous test imgs) |

### 3.4 Crop definition (ECF patch-based)
- Crop size: 256×256, centred on the defect mask centroid.
- If the defect bounding box exceeds 256, fall back to the tightest square that contains it, then resize to 256.
- If the defect sits near an image edge, clamp the crop window inside the image and record the offset in metadata so the composite can place it back exactly.
- The crop's local mask is the corresponding region of the full mask.

### 3.5 Output contract (what C2 produces for downstream)
Each synthetic sample is a triple written atomically:
```
outputs/synthetic/{dataset}/{class}/images/{uid}.png   # RGB synthetic defect image
outputs/synthetic/{dataset}/{class}/masks/{uid}.png    # binary mask (label)
outputs/synthetic/{dataset}/{class}/meta/{uid}.json    # provenance
```
`meta/{uid}.json` schema:
```json
{
  "uid": "string",
  "dataset": "mvtec_ad | ecf",
  "class": "string",
  "source_clean_image": "path",
  "source_mask": "path",
  "generation_mode": "whole | patch",
  "crop_offset": [x, y] ,
  "lora_checkpoint": "path",
  "token": "<class-token>",
  "seed": 0,
  "num_inference_steps": 50,
  "guidance_scale": 7.5,
  "lpips_score": 0.0,
  "lfs_passed": true
}
```

---

## 4. Mathematical Formulation

Notation: $\epsilon_\theta$ is the UNet noise predictor with LoRA-adapted weights; $x_t$ the noised latent at timestep $t$; $M$ the binary defect mask resampled to latent resolution; $\odot$ elementwise product; $c$ the text condition built from the learned defect token; $\mathcal{E}, \mathcal{D}$ the frozen VAE encoder/decoder; $y$ a clean image; $m$ the pixel-space mask; $\hat{x}$ the decoded generated image.

### 4.1 Defect loss
Concentrates capacity on the defect texture by computing the denoising objective **only inside the mask**:

$$
\mathcal{L}_{\text{def}} = \mathbb{E}_{x,t,\epsilon}\!\left[\; \big\| M \odot \big(\epsilon - \epsilon_\theta(x_t^{\text{def}}, t, c^{\text{def}})\big) \big\|_2^2 \;\right]
$$

### 4.2 Object loss
Preserves the semantic relationship between the defect and the surrounding object so the fill blends naturally. Computed over the whole image but with the background down-weighted by $\alpha \in (0,1)$:

$$
\mathcal{L}_{\text{obj}} = \mathbb{E}_{x,t,\epsilon}\!\left[\; \big\| M' \odot \big(\epsilon - \epsilon_\theta(x_t^{\text{obj}}, t, c^{\text{obj}})\big) \big\|_2^2 \;\right],
\qquad
M' = M + \alpha\,(1 - M)
$$

### 4.3 Attention loss
Binds the learned defect token to the defect region by aligning the token's cross-attention map $A_{\text{token}}$ with the mask:

$$
\mathcal{L}_{\text{attn}} = \mathbb{E}\!\left[\; \big\| A_{\text{token}} - M \big\|_2^2 \;\right]
$$

### 4.4 Total objective

$$
\mathcal{L}_{\text{C2}} = \mathcal{L}_{\text{def}} + \lambda_{\text{obj}}\,\mathcal{L}_{\text{obj}} + \lambda_{\text{attn}}\,\mathcal{L}_{\text{attn}}
$$

Default weights: $\lambda_{\text{obj}} = 1.0$, $\lambda_{\text{attn}} = 0.1$, $\alpha = 0.1$. All three are config-exposed and are ablation variables.

### 4.5 Latent-space compositing (inference, per denoising step)
At each timestep the known region is forcibly reasserted so the model can only change the masked region:

$$
z_t \leftarrow M \odot z_t^{\text{denoised}} + (1 - M) \odot z_t^{\text{known}}
$$

where $z_t^{\text{known}}$ is the clean latent $\mathcal{E}(y)$ noised to timestep $t$.

### 4.6 Pixel-space compositing (post-hoc, anti-aliasing safeguard)
After decoding, a final hard composite in pixel space eliminates any residual VAE bleed at the mask boundary — critical for micro-defects:

$$
\hat{x}_{\text{final}} = m \odot \hat{x} + (1 - m) \odot y
$$

This guarantees every non-defect pixel is bit-identical to the real clean image.

### 4.7 Low-Fidelity Selection score
For a generated image $\hat{x}$ with clean source $y$ and mask $m$, the LFS score is the masked perceptual distance:

$$
s_{\text{LFS}} = \text{LPIPS}\big(m \odot \hat{x},\; m \odot y\big)
$$

A higher score means a more pronounced (more clearly expressed) defect. Samples below a per-class percentile threshold $\tau$ (default: 25th percentile) are rejected.

---

## 5. Hardware, VRAM, and Windows Protections

### 5.1 Precision
- **bf16 everywhere** for UNet forward/backward (`mixed_precision="bf16"` in accelerate). Native on Blackwell; more stable than fp16, no loss scaling.
- **VAE decode safeguard:** decode in fp32, or validate bf16 decodes produce no black/NaN images during Phase 0. If any NaN is detected, force `pipe.vae.to(torch.float32)` for decode. This check is mandatory in Phase 0's acceptance test.

### 5.2 Attention
- Use PyTorch native **SDPA** (`scaled_dot_product_attention`). Do **not** depend on xFormers (Windows build fragility). Attention slicing only if memory is ever tight (it will not be at these sizes).

### 5.3 Windows DataLoader rule (mandatory, non-negotiable)
**Every** `torch.utils.data.DataLoader` in C2 must set `num_workers=0`. Windows uses `spawn`, and worker processes caused GPU-starvation deadlocks in C1. This applies to training, generation batching, and downstream training. A repo-wide grep in `verify_phase_6.py` asserts no `num_workers=` with a nonzero value exists in `src/c2_synthesis/`.

### 5.4 Memory levers (mostly headroom, documented for SDXL ablation)
- SD 1.5 inpainting LoRA at 512px trains in <12 GB of 48 GB — vast headroom.
- Gradient checkpointing / gradient accumulation / 8-bit Adam / CPU offload are available but **off by default**; enable only for an optional SDXL ablation. If `bitsandbytes` fails to import on Windows, fall back to AdamW (documented).

### 5.5 Determinism
- Global seed set via `utils/seed.py` (`random`, `numpy`, `torch`, `torch.cuda`, `PYTHONHASHSEED`).
- Generation records its seed per sample in metadata.
- CUDA determinism flags set where they do not cripple throughput.

### 5.6 Long-run stability
- Disable Windows sleep during batch generation / downstream sweeps.
- Every long phase runs a **timed smoke test** (one class, few steps) before the full run — the direct lesson from the C1 Dinomaly time-estimate error. Never assume throughput; measure it.

---

## 6. Artifact Storage and Versioning Protocols

### 6.1 Storage matrix

| Artifact | Location | git | DVC | MLflow |
|---|---|---|---|---|
| LoRA weights, token embeddings | `outputs/checkpoints/c2/` | no | yes | path param |
| Synthetic images/masks/meta | `outputs/synthetic/` | no | yes | no |
| Fidelity/downstream/ablation CSVs | `outputs/tables/c2/` | yes | no | metrics |
| Figures (300 dpi) | `outputs/figures/c2/` | yes | no | no |
| Hyperparameters, metrics | — | no | no | yes |

Rule: small + human-readable → git; large + binary → DVC; metrics + params → MLflow.

### 6.2 MLflow schema
- Experiments: `c2-train-{dataset}`, `c2-downstream-{dataset}`, `c2-ablation-{name}`.
- Run naming: `{class}-r{rank}-{steps}steps` (train); `{config}-b{budget}-seed{n}` (downstream).
- Logged params: base model, rank, alpha, loss weights ($\lambda_{obj},\lambda_{attn},\alpha$), steps, lr, seed, dataset, class, generation config.
- Logged metrics: training loss terms; FID/KID/LPIPS (fidelity); I-AUROC, AP, P-AUROC, AUPRO, pixel-F1 (downstream); LFS acceptance rate; wall time; peak VRAM.

### 6.3 DVC protocol
After each artifact-producing phase:
```
dvc add outputs/checkpoints/c2 outputs/synthetic
git add outputs/checkpoints/c2.dvc outputs/synthetic.dvc
git commit -m "C2 <phase>: <summary>"
git tag c2-<milestone>
```

### 6.4 Config-driven everything
No magic numbers in code. All hyperparameters live in `src/c2_synthesis/configs/*.yaml`, loaded at entry points. Changing an experiment means editing a YAML, not source.

---

## 7. Acceptance Philosophy

Every phase ends with a `tests/verify_phase_X.py` that must pass before the next phase begins. A phase is "done" only when its verification script exits 0 and its stated gate artifact exists. This mirrors the C1 discipline: verify at every step, never scale on an unverified assumption. The execution plan (`C2_EXECUTION_PLAN.md`) specifies each phase's exact acceptance test.
