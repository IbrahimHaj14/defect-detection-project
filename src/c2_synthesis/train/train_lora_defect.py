"""Phase 2 LoRA + learned-token training for one C2 pilot class."""

from __future__ import annotations

import argparse
import gc
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import mlflow
import numpy as np
import torch
import yaml
from diffusers import DDIMScheduler, StableDiffusionInpaintPipeline
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from src.c2_synthesis.data.pair_builder import build_pairs, load_dataset_config
from src.c2_synthesis.data.patch_extractor import extract_defect_crop
from src.c2_synthesis.train.losses import C2Losses, defect_loss, object_loss, total_c2_loss
from src.c2_synthesis.train.token_manager import LearnedTokenManager
from src.c2_synthesis.utils.image_io import load_image_rgb, load_mask_binary
from src.c2_synthesis.utils.logging_utils import get_logger
from src.c2_synthesis.utils.mlflow_utils import start_c2_run
from src.c2_synthesis.utils.seed import set_global_seed

logger = get_logger(__name__)

_C2_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _C2_ROOT / "configs" / "lora_defect.yaml"
_SUPPORTED_DTYPES = {"bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class TrainingResult:
    """Artifacts, measurements, and step losses from one training stage."""

    history: tuple[dict[str, float], ...]
    lora_path: Path
    token_path: Path
    steps_per_second: float
    projected_1000_seconds: float
    elapsed_seconds: float
    peak_vram_gib: float


def _repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    return path if path.is_absolute() else _REPO_ROOT / path


def load_training_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the complete Phase 2 training contract."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    required = {
        "base_model_id",
        "dataset_key",
        "class_name",
        "pair_budget",
        "resolution",
        "learned_token",
        "defect_prompt_template",
        "object_prompt_template",
        "rank",
        "lora_alpha",
        "target_modules",
        "learning_rate",
        "max_steps",
        "smoke_steps",
        "max_projected_pilot_minutes",
        "batch_size",
        "gradient_accumulation",
        "lambda_obj",
        "lambda_attn",
        "object_background_alpha",
        "object_mask_min_bbox_area_scale",
        "object_mask_max_bbox_area_scale",
        "seed",
        "num_workers",
        "dtype",
        "checkpoint_root",
    }
    if not isinstance(config, dict):
        raise ValueError(f"Phase 2 config must be a mapping: {config_path}")
    # Backward compatibility keeps historical baseline/calibration YAMLs valid
    # and disabled unless the ablation flag is explicitly enabled.
    config.setdefault("enable_masked_ti", False)
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Phase 2 config is missing keys: {sorted(missing)}")

    if int(config["num_workers"]) != 0:
        raise ValueError("C2 DataLoaders must use num_workers=0 on Windows")
    if int(config["batch_size"]) != 1 or int(config["gradient_accumulation"]) != 1:
        raise ValueError("Phase 2 pilot requires batch_size=1 and gradient_accumulation=1")
    if int(config["pair_budget"]) <= 0:
        raise ValueError("pair_budget must be positive")
    if int(config["smoke_steps"]) <= 0:
        raise ValueError("smoke_steps must be positive")
    if int(config["max_steps"]) < int(config["smoke_steps"]):
        raise ValueError("max_steps must be greater than or equal to smoke_steps")
    if float(config["max_projected_pilot_minutes"]) <= 0.0:
        raise ValueError("max_projected_pilot_minutes must be positive")
    if str(config["dtype"]) not in _SUPPORTED_DTYPES:
        raise ValueError("Phase 2 supports native bfloat16 training only")
    if not isinstance(config["enable_masked_ti"], bool):
        raise ValueError("enable_masked_ti must be a boolean")
    min_scale = float(config["object_mask_min_bbox_area_scale"])
    max_scale = float(config["object_mask_max_bbox_area_scale"])
    if not 1.0 <= min_scale <= max_scale:
        raise ValueError("Object-mask area scales must satisfy 1 <= min <= max")
    for template_key in ("defect_prompt_template", "object_prompt_template"):
        if "{token}" not in str(config[template_key]):
            raise ValueError(f"{template_key} must contain {{token}}")
    return config


def make_object_rectangle_mask(
    defect_mask: np.ndarray,
    *,
    min_bbox_area_scale: float,
    max_bbox_area_scale: float,
    rng: random.Random,
) -> np.ndarray:
    """Sample an in-frame rectangle covering the defect under area bounds.

    The rectangle always contains the full defect bounding box. Its integer
    area is at least ``min_bbox_area_scale`` and at most
    ``max_bbox_area_scale`` times the bounding-box area.
    """

    binary = np.asarray(defect_mask) > 0
    if binary.ndim != 2:
        raise ValueError("defect_mask must be a two-dimensional array")
    ys, xs = np.nonzero(binary)
    if xs.size == 0:
        raise ValueError("Cannot construct an object mask from an empty defect mask")
    if not 1.0 <= float(min_bbox_area_scale) <= float(max_bbox_area_scale):
        raise ValueError("Mask area scales must satisfy 1 <= min <= max")

    height, width = binary.shape
    bbox_x0, bbox_x1 = int(xs.min()), int(xs.max()) + 1
    bbox_y0, bbox_y1 = int(ys.min()), int(ys.max()) + 1
    bbox_width = bbox_x1 - bbox_x0
    bbox_height = bbox_y1 - bbox_y0
    bbox_area = bbox_width * bbox_height
    min_area = math.ceil(float(min_bbox_area_scale) * bbox_area)
    max_area = min(
        width * height,
        math.floor(float(max_bbox_area_scale) * bbox_area),
    )

    candidates: list[tuple[int, int, int]] = []
    for rectangle_width in range(bbox_width, width + 1):
        minimum_height = max(bbox_height, math.ceil(min_area / rectangle_width))
        maximum_height = min(height, max_area // rectangle_width)
        if minimum_height <= maximum_height:
            candidates.append((rectangle_width, minimum_height, maximum_height))
    if not candidates:
        feasible_scale = (width * height) / bbox_area
        raise ValueError(
            "No in-frame rectangular mask satisfies the configured area bounds; "
            f"maximum feasible area scale is {feasible_scale:.3f}"
        )

    rectangle_width, min_height, max_height = rng.choice(candidates)
    rectangle_height = rng.randint(min_height, max_height)
    min_x = max(0, bbox_x1 - rectangle_width)
    max_x = min(bbox_x0, width - rectangle_width)
    min_y = max(0, bbox_y1 - rectangle_height)
    max_y = min(bbox_y0, height - rectangle_height)
    if min_x > max_x or min_y > max_y:
        raise RuntimeError("Internal error while placing a defect-containing rectangle")
    rectangle_x = rng.randint(min_x, max_x)
    rectangle_y = rng.randint(min_y, max_y)

    object_mask = np.zeros_like(binary, dtype=np.uint8)
    object_mask[
        rectangle_y : rectangle_y + rectangle_height,
        rectangle_x : rectangle_x + rectangle_width,
    ] = 1
    return object_mask


def _resize_training_pair(
    image_path: str,
    mask_path: str,
    resolution: int,
    *,
    generation_mode: str,
    crop_size: int | None,
) -> tuple[Tensor, np.ndarray]:
    image = load_image_rgb(_repo_path(image_path))
    source_mask = load_mask_binary(_repo_path(mask_path))
    if generation_mode == "patch":
        if crop_size is None:
            raise ValueError("Patch-based training requires a configured crop_size")
        crop = extract_defect_crop(image, source_mask, size=int(crop_size))
        if crop is None:
            raise ValueError(f"Defect pair is not crop-eligible: {mask_path}")
        image = crop.image
        source_mask = crop.mask
    elif generation_mode != "whole":
        raise ValueError(f"Unsupported training generation_mode: {generation_mode}")

    image = image.resize((resolution, resolution), resample=Image.Resampling.BICUBIC)
    resized_mask = np.asarray(
        Image.fromarray(source_mask * 255).resize(
            (resolution, resolution),
            resample=Image.Resampling.NEAREST,
        ),
        dtype=np.uint8,
    )
    resized_mask = (resized_mask > 127).astype(np.uint8)
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
    pixels = pixels.permute(2, 0, 1).div(127.5).sub(1.0)
    return pixels, resized_mask


def _encode_vae_latents(vae: torch.nn.Module, pixels: Tensor) -> Tensor:
    posterior = vae.encode(pixels).latent_dist
    return posterior.mode() * float(vae.config.scaling_factor)


def _latent_mask(mask: Tensor, latent_size: tuple[int, int]) -> Tensor:
    """Resample with max pooling so latent-space micro-defects cannot vanish."""

    return F.adaptive_max_pool2d(mask.float(), latent_size).clamp(0.0, 1.0)


def _prepare_latent_examples(
    pipe: StableDiffusionInpaintPipeline,
    pairs: Sequence[tuple[str, str]],
    config: Mapping[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, Tensor]]:
    resolution = int(config["resolution"])
    dataset_config = load_dataset_config(str(config["dataset_key"]))
    generation_mode = str(dataset_config["generation_mode"])
    crop_size_value = dataset_config.get("crop_size")
    crop_size = None if crop_size_value is None else int(crop_size_value)
    rng = random.Random(int(config["seed"]))
    noise_generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    examples: list[dict[str, Tensor]] = []
    num_train_timesteps = int(pipe.scheduler.config.num_train_timesteps)

    pipe.vae.eval()
    with torch.no_grad():
        for image_path, mask_path in pairs:
            pixels_cpu, defect_mask_array = _resize_training_pair(
                image_path,
                mask_path,
                resolution,
                generation_mode=generation_mode,
                crop_size=crop_size,
            )
            object_mask_array = make_object_rectangle_mask(
                defect_mask_array,
                min_bbox_area_scale=float(
                    config["object_mask_min_bbox_area_scale"]
                ),
                max_bbox_area_scale=float(
                    config["object_mask_max_bbox_area_scale"]
                ),
                rng=rng,
            )
            pixels = pixels_cpu.unsqueeze(0).to(device=device, dtype=dtype)
            defect_mask = torch.from_numpy(defect_mask_array).unsqueeze(0).unsqueeze(0)
            object_mask = torch.from_numpy(object_mask_array).unsqueeze(0).unsqueeze(0)
            defect_mask = defect_mask.to(device=device, dtype=dtype)
            object_mask = object_mask.to(device=device, dtype=dtype)

            target_latent = _encode_vae_latents(pipe.vae, pixels)
            defect_masked_latent = _encode_vae_latents(
                pipe.vae,
                pixels * (1.0 - defect_mask),
            )
            object_masked_latent = _encode_vae_latents(
                pipe.vae,
                pixels * (1.0 - object_mask),
            )
            latent_size = tuple(int(value) for value in target_latent.shape[-2:])
            defect_mask_latent = _latent_mask(defect_mask, latent_size)
            object_mask_latent = _latent_mask(object_mask, latent_size)
            noise = torch.randn(
                target_latent.shape,
                generator=noise_generator,
                device=device,
                dtype=dtype,
            )
            timestep = torch.randint(
                0,
                num_train_timesteps,
                (1,),
                generator=noise_generator,
                device=device,
                dtype=torch.int64,
            )

            examples.append(
                {
                    "target_latent": target_latent.squeeze(0).cpu(),
                    "defect_masked_latent": defect_masked_latent.squeeze(0).cpu(),
                    "object_masked_latent": object_masked_latent.squeeze(0).cpu(),
                    "defect_mask_latent": defect_mask_latent.squeeze(0).cpu(),
                    "object_mask_latent": object_mask_latent.squeeze(0).cpu(),
                    "noise": noise.squeeze(0).cpu(),
                    "timestep": timestep.squeeze(0).cpu(),
                }
            )
    return examples


def _expand_cross_attention_targets(
    unet: torch.nn.Module,
    requested_targets: Sequence[str],
) -> list[str]:
    """Resolve short projection names to decoder and encoder ``attn2`` paths."""

    requested = tuple(str(target) for target in requested_targets)
    exact_targets = [
        name
        for name, _ in unet.named_modules()
        if ".attn2." in name
        and any(name.endswith(f".{target}") for target in requested)
    ]
    if not exact_targets:
        raise RuntimeError(
            f"No UNet cross-attention modules matched configured targets {requested}"
        )
    if any(".attn1." in name for name in exact_targets):
        raise RuntimeError("LoRA target expansion unexpectedly included self-attention")
    return exact_targets


def inject_cross_attention_lora(
    unet: torch.nn.Module,
    *,
    rank: int,
    alpha: int,
    target_modules: Sequence[str],
) -> list[str]:
    """Inject PEFT LoRA into only the UNet cross-attention projections."""

    exact_targets = _expand_cross_attention_targets(unet, target_modules)
    lora_config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        target_modules=exact_targets,
        init_lora_weights=True,
        bias="none",
    )
    unet.add_adapter(lora_config)
    return exact_targets


class TokenAttentionCollector:
    """Collect differentiable learned-token maps from decoder cross-attention."""

    def __init__(
        self,
        unet: torch.nn.Module,
        latent_size: tuple[int, int],
    ) -> None:
        self.latent_size = latent_size
        self.enabled = False
        self.token_positions: Tensor | None = None
        self.maps: list[Tensor] = []
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        for name, module in unet.named_modules():
            if name.startswith("up_blocks.") and name.endswith(".attn2"):
                self.handles.append(
                    module.register_forward_hook(self._capture, with_kwargs=True)
                )
        if not self.handles:
            raise RuntimeError("No decoder cross-attention modules found for attention loss")

    def begin(self, token_positions: Tensor) -> None:
        self.maps.clear()
        self.token_positions = token_positions
        self.enabled = True

    def end(self) -> Tensor | None:
        self.enabled = False
        self.token_positions = None
        if not self.maps:
            return None
        return torch.stack(self.maps, dim=0).mean(dim=0)

    def _capture(
        self,
        module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        _output: Any,
    ) -> None:
        if not self.enabled or self.token_positions is None:
            return
        try:
            hidden_states = args[0]
            encoder_hidden_states = kwargs.get("encoder_hidden_states")
            if encoder_hidden_states is None and len(args) > 1:
                encoder_hidden_states = args[1]
            if hidden_states.ndim != 3 or encoder_hidden_states is None:
                raise ValueError("cross-attention inputs are not three-dimensional")

            # Native SDPA intentionally remains active for the real UNet forward.
            # SDPA does not materialise intermediate attention probabilities, so
            # this hook separately reconstructs differentiable Q.K^T.softmax for
            # the learned token only. This avoids xFormers/eager-attention Windows
            # regressions while still providing the DefectFill attention loss.
            query = module.to_q(hidden_states)
            key = module.to_k(encoder_hidden_states)
            batch, query_length, inner_dim = query.shape
            heads = int(module.heads)
            head_dim = inner_dim // heads
            query = query.view(batch, query_length, heads, head_dim).transpose(1, 2)
            key = key.view(batch, key.shape[1], heads, head_dim).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-1, -2)) * float(module.scale)
            probabilities = scores.softmax(dim=-1)
            positions = self.token_positions.to(device=probabilities.device)
            if positions.shape != (batch,) or bool((positions >= key.shape[2]).any().item()):
                raise ValueError("learned-token positions do not match attention keys")
            gather_index = positions.view(batch, 1, 1, 1).expand(
                batch,
                heads,
                query_length,
                1,
            )
            token_map = probabilities.gather(-1, gather_index).squeeze(-1).mean(dim=1)
            spatial_side = math.isqrt(query_length)
            if spatial_side * spatial_side != query_length:
                raise ValueError(f"attention query length {query_length} is not square")
            token_map = token_map.view(batch, 1, spatial_side, spatial_side)
            if token_map.shape[-2:] != self.latent_size:
                token_map = F.interpolate(
                    token_map,
                    size=self.latent_size,
                    mode="bilinear",
                    align_corners=False,
                )
            self.maps.append(token_map)
        except (AttributeError, IndexError, RuntimeError, TypeError, ValueError) as error:
            logger.warning("Skipping incompatible decoder attention map: %s", error)

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def _move_batch(
    batch: Mapping[str, Tensor],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor]:
    moved: dict[str, Tensor] = {}
    for key, value in batch.items():
        target_dtype = torch.int64 if key == "timestep" else dtype
        moved[key] = value.to(device=device, dtype=target_dtype, non_blocking=False)
    return moved


def _cycle(loader: Iterable[Mapping[str, Tensor]]) -> Iterable[Mapping[str, Tensor]]:
    while True:
        yield from loader


def _format_prompt(template: str, *, token: str, object_name: str) -> str:
    try:
        return template.format(token=token, object=object_name)
    except KeyError as error:
        raise ValueError(f"Unsupported prompt-template placeholder: {error}") from error


def _save_lora(unet: torch.nn.Module, path: Path) -> Path:
    state = get_peft_model_state_dict(unet)
    if not state:
        raise RuntimeError("PEFT returned an empty LoRA state dictionary")
    serialisable = {
        key: value.detach().contiguous().cpu() for key, value in state.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(serialisable, path, metadata={"format": "pt", "component": "c2-unet-lora"})
    return path


def load_lora_checkpoint(unet: torch.nn.Module, path: str | Path) -> None:
    """Load a Phase 2 safetensors checkpoint into an already-adapted UNet."""

    state = load_file(str(path), device="cpu")
    if not state:
        raise ValueError(f"LoRA checkpoint contains no tensors: {path}")
    incompatible = set_peft_model_state_dict(unet, state, adapter_name="default")
    unexpected = tuple(getattr(incompatible, "unexpected_keys", ()))
    if unexpected:
        raise ValueError(f"Unexpected LoRA keys while loading {path}: {unexpected}")


def _finite_loss_record(step: int, losses: C2Losses) -> dict[str, float]:
    record = {
        "step": float(step),
        "loss_total": float(losses.total.detach().float().item()),
        "loss_defect": float(losses.defect.detach().float().item()),
        "loss_object": float(losses.object.detach().float().item()),
        "loss_attention": float(losses.attention.detach().float().item()),
    }
    if not all(math.isfinite(value) for key, value in record.items() if key != "step"):
        raise FloatingPointError(f"Non-finite Phase 2 loss at step {step}: {record}")
    return record


def backward_c2_losses(
    losses: C2Losses,
    *,
    lora_parameters: Sequence[torch.nn.Parameter],
    token_parameter: torch.nn.Parameter,
    enable_masked_ti: bool,
) -> None:
    """Backpropagate full LoRA loss and optional defect-only token loss.

    With masked textual inversion enabled, the LoRA adapters still receive the
    complete three-term DefectFill objective. The learned token receives only
    ``losses.defect``, whose masked-pixel normalization restricts its learning
    signal to the defect support. This follows the background-leakage
    suppression described by the NPI defect-synthesis paper
    (arXiv:2604.22850, masked textual inversion in Section 3).
    """

    if not enable_masked_ti:
        losses.total.backward()
        return
    if not lora_parameters:
        raise ValueError("Masked textual inversion requires LoRA parameters")
    # Restrict the first traversal to LoRA leaves, then traverse the retained
    # defect graph only for the embedding table. The token manager's gradient
    # hook continues to zero every vocabulary row except the learned token.
    torch.autograd.backward(
        losses.total,
        inputs=tuple(lora_parameters),
        retain_graph=True,
    )
    torch.autograd.backward(losses.defect, inputs=(token_parameter,))


def train_lora_defect(
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    steps: int | None = None,
    enforce_smoke_gate: bool = True,
    log_to_mlflow: bool = True,
    pairs_override: Sequence[tuple[str, str]] | None = None,
) -> TrainingResult:
    """Train one actual C2 LoRA/token adapter and write spec-compliant artifacts.

    ``pairs_override`` is reserved for the Phase 4 sweep, which filters the
    deterministic manifest order through the approved ECF crop and object-mask
    eligibility rules before choosing its five production pairs. Omitting it
    preserves the Phase 2 pairing contract exactly.
    """

    config = load_training_config(config_path)
    training_steps = int(config["max_steps"] if steps is None else steps)
    if training_steps <= 0:
        raise ValueError("training steps must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Phase 2 requires a CUDA GPU with native bfloat16 support")

    seed = int(config["seed"])
    set_global_seed(seed)
    device = torch.device("cuda")
    dtype = _SUPPORTED_DTYPES[str(config["dtype"])]
    if pairs_override is None:
        pairs, _clean_images = build_pairs(
            str(config["dataset_key"]),
            str(config["class_name"]),
            int(config["pair_budget"]),
        )
    else:
        pairs = list(pairs_override)
    if len(pairs) != int(config["pair_budget"]):
        raise RuntimeError(
            f"Training requested {config['pair_budget']} pairs but received {len(pairs)}"
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        str(config["base_model_id"]),
        torch_dtype=dtype,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.requires_grad_(False).to(device=device, dtype=dtype)
    pipe.unet.requires_grad_(False).to(device=device, dtype=dtype)
    pipe.text_encoder.requires_grad_(False).to(device=device, dtype=torch.float32)

    examples = _prepare_latent_examples(pipe, pairs, config, device, dtype)
    loader = DataLoader(
        examples,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
    )
    if loader.num_workers != 0:
        raise RuntimeError("Windows-safe C2 DataLoader unexpectedly has workers")

    token_manager = LearnedTokenManager(
        pipe.tokenizer,
        pipe.text_encoder,
        str(config["learned_token"]),
    )
    exact_lora_targets = inject_cross_attention_lora(
        pipe.unet,
        rank=int(config["rank"]),
        alpha=int(config["lora_alpha"]),
        target_modules=list(config["target_modules"]),
    )
    pipe.unet.train()
    lora_parameters = [parameter for parameter in pipe.unet.parameters() if parameter.requires_grad]
    if not lora_parameters:
        raise RuntimeError("LoRA injection produced no trainable UNet parameters")

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_parameters, "weight_decay": 1.0e-2},
            {"params": [token_manager.trainable_parameter], "weight_decay": 0.0},
        ],
        lr=float(config["learning_rate"]),
    )
    latent_size = tuple(int(value) for value in examples[0]["target_latent"].shape[-2:])
    attention_collector = TokenAttentionCollector(pipe.unet, latent_size)
    defect_prompt = _format_prompt(
        str(config["defect_prompt_template"]),
        token=token_manager.token,
        object_name=str(config["class_name"]).replace("_", " "),
    )
    object_prompt = _format_prompt(
        str(config["object_prompt_template"]),
        token=token_manager.token,
        object_name=str(config["class_name"]).replace("_", " "),
    )

    checkpoint_dir = (
        _repo_path(str(config["checkpoint_root"]))
        / str(config["dataset_key"])
        / str(config["class_name"])
    )
    lora_path = checkpoint_dir / "lora.safetensors"
    token_path = checkpoint_dir / "token.pt"
    run_params = {
        "base_model": config["base_model_id"],
        "rank": config["rank"],
        "lora_alpha": config["lora_alpha"],
        "lora_target_count": len(exact_lora_targets),
        "lambda_obj": config["lambda_obj"],
        "lambda_attn": config["lambda_attn"],
        "object_background_alpha": config["object_background_alpha"],
        "steps": training_steps,
        "lr": config["learning_rate"],
        "seed": config["seed"],
        "dataset": config["dataset_key"],
        "class": config["class_name"],
        "pair_budget": config["pair_budget"],
        "enable_masked_ti": config["enable_masked_ti"],
        "lora_checkpoint": lora_path,
        "token_checkpoint": token_path,
    }

    history: list[dict[str, float]] = []
    started_at = time.perf_counter()

    def execute_steps() -> None:
        for step_index, raw_batch in zip(range(training_steps), _cycle(loader), strict=False):
            batch = _move_batch(raw_batch, device, dtype)
            optimizer.zero_grad(set_to_none=True)
            defect_hidden, token_positions = token_manager.encode_prompt(
                defect_prompt,
                device=device,
            )
            object_hidden, _ = token_manager.encode_prompt(object_prompt, device=device)
            defect_hidden = defect_hidden.to(dtype=dtype)
            object_hidden = object_hidden.to(dtype=dtype)
            noisy_latents = pipe.scheduler.add_noise(
                batch["target_latent"],
                batch["noise"],
                batch["timestep"],
            )

            attention_collector.begin(token_positions)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                defect_input = torch.cat(
                    [
                        noisy_latents,
                        batch["defect_mask_latent"],
                        batch["defect_masked_latent"],
                    ],
                    dim=1,
                )
                defect_prediction = pipe.unet(
                    defect_input,
                    batch["timestep"],
                    encoder_hidden_states=defect_hidden,
                    return_dict=False,
                )[0]
            attention_map = attention_collector.end()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                object_input = torch.cat(
                    [
                        noisy_latents,
                        batch["object_mask_latent"],
                        batch["object_masked_latent"],
                    ],
                    dim=1,
                )
                object_prediction = pipe.unet(
                    object_input,
                    batch["timestep"],
                    encoder_hidden_states=object_hidden,
                    return_dict=False,
                )[0]

            if attention_map is None:
                logger.warning(
                    "Step %d has no compatible attention maps; skipping attention loss",
                    step_index + 1,
                )
                loss_defect = defect_loss(
                    batch["noise"], defect_prediction, batch["defect_mask_latent"]
                )
                loss_object = object_loss(
                    batch["noise"],
                    object_prediction,
                    batch["object_mask_latent"],
                    float(config["object_background_alpha"]),
                )
                zero_attention = loss_defect.new_zeros(())
                losses = C2Losses(
                    total=loss_defect + float(config["lambda_obj"]) * loss_object,
                    defect=loss_defect,
                    object=loss_object,
                    attention=zero_attention,
                )
            else:
                losses = total_c2_loss(
                    batch["noise"],
                    defect_prediction,
                    object_prediction,
                    batch["defect_mask_latent"],
                    batch["object_mask_latent"],
                    attention_map,
                    alpha=float(config["object_background_alpha"]),
                    lambda_obj=float(config["lambda_obj"]),
                    lambda_attn=float(config["lambda_attn"]),
                )
            backward_c2_losses(
                losses,
                lora_parameters=lora_parameters,
                token_parameter=token_manager.trainable_parameter,
                enable_masked_ti=bool(config["enable_masked_ti"]),
            )
            optimizer.step()

            record = _finite_loss_record(step_index + 1, losses)
            history.append(record)
            if log_to_mlflow:
                mlflow.log_metrics(
                    {key: value for key, value in record.items() if key != "step"},
                    step=step_index + 1,
                )
            logger.info(
                "step=%d/%d total=%.6f defect=%.6f object=%.6f attention=%.6f",
                step_index + 1,
                training_steps,
                record["loss_total"],
                record["loss_defect"],
                record["loss_object"],
                record["loss_attention"],
            )

    try:
        if log_to_mlflow:
            experiment = f"c2-train-{config['dataset_key']}"
            run_name = f"{config['class_name']}-r{config['rank']}-{training_steps}steps"
            with start_c2_run(experiment, run_name, run_params):
                execute_steps()
        else:
            execute_steps()

        torch.cuda.synchronize(device)
        elapsed_seconds = time.perf_counter() - started_at
        steps_per_second = training_steps / elapsed_seconds
        projected_seconds = int(config["max_steps"]) / steps_per_second
        projected_minutes = projected_seconds / 60.0
        print(f"Training throughput: {steps_per_second:.4f} steps/s")
        print(
            f"Projected {int(config['max_steps'])}-step wall time: "
            f"{projected_minutes:.2f} minutes"
        )
        if (
            enforce_smoke_gate
            and training_steps == int(config["smoke_steps"])
            and projected_minutes > float(config["max_projected_pilot_minutes"])
        ):
            raise RuntimeError(
                "Phase 2 pilot aborted: the timed 50-step smoke test projects "
                f"{projected_minutes:.2f} minutes for {config['max_steps']} steps, "
                f"exceeding the approved {config['max_projected_pilot_minutes']} minute limit."
            )

        _save_lora(pipe.unet, lora_path)
        token_manager.save(token_path)
        peak_vram_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        return TrainingResult(
            history=tuple(history),
            lora_path=lora_path,
            token_path=token_path,
            steps_per_second=steps_per_second,
            projected_1000_seconds=projected_seconds,
            elapsed_seconds=elapsed_seconds,
            peak_vram_gib=peak_vram_gib,
        )
    finally:
        attention_collector.close()
        token_manager.close()


def _release_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--stage",
        choices=("smoke", "pilot"),
        default="smoke",
        help="'pilot' always runs the mandatory 50-step gate before 1000 steps",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow only for local diagnostics",
    )
    args = parser.parse_args()
    config = load_training_config(args.config)
    smoke_result = train_lora_defect(
        args.config,
        steps=int(config["smoke_steps"]),
        enforce_smoke_gate=True,
        log_to_mlflow=not args.no_mlflow,
    )
    print(f"Smoke artifacts: {smoke_result.lora_path.parent.as_posix()}")
    if args.stage == "pilot":
        _release_cuda()
        pilot_result = train_lora_defect(
            args.config,
            steps=int(config["max_steps"]),
            enforce_smoke_gate=False,
            log_to_mlflow=not args.no_mlflow,
        )
        print(f"Pilot artifacts: {pilot_result.lora_path.parent.as_posix()}")


if __name__ == "__main__":
    main()
