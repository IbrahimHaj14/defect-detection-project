"""Phase 2 acceptance test: actual 50-step LoRA + token smoke training."""

from __future__ import annotations

import gc
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers import UNet2DConditionModel
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c2_synthesis.train.losses import (  # noqa: E402
    attention_loss,
    defect_loss,
    object_loss,
    total_c2_loss,
)
from src.c2_synthesis.train.token_manager import LearnedTokenManager  # noqa: E402
from src.c2_synthesis.train.train_lora_defect import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    inject_cross_attention_lora,
    load_lora_checkpoint,
    load_training_config,
    make_object_rectangle_mask,
    train_lora_defect,
)

VERIFY_CONFIG_PATH = REPO_ROOT / "outputs/logs/c2/phase2_verify_config.yaml"
VERIFY_CHECKPOINT_ROOT = "outputs/checkpoints/c2_phase2_verify"


def _isolated_verify_config(config: dict[str, object]) -> tuple[Path, dict[str, object]]:
    """Keep the acceptance smoke from overwriting the 1000-step baseline."""

    isolated = dict(config)
    isolated["checkpoint_root"] = VERIFY_CHECKPOINT_ROOT
    VERIFY_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with VERIFY_CONFIG_PATH.open("w", encoding="utf-8", newline="\n") as config_file:
        yaml.safe_dump(isolated, config_file, sort_keys=False)
    return VERIFY_CONFIG_PATH, isolated


def _verify_loss_equations() -> None:
    noise = torch.ones(1, 4, 8, 8, dtype=torch.float32)
    prediction = torch.zeros_like(noise)
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.float32)
    mask[:, :, 2:4, 3:5] = 1.0

    measured_defect = defect_loss(noise, prediction, mask)
    assert torch.isclose(measured_defect, torch.tensor(1.0)), (
        "defect_loss is not normalised over masked pixels only"
    )
    expected_object = (4.0 + 60.0 * 0.1**2) / 64.0
    measured_object = object_loss(noise, prediction, mask, alpha=0.1)
    assert torch.isclose(measured_object, torch.tensor(expected_object)), (
        f"object_loss equation mismatch: {measured_object.item()} != {expected_object}"
    )
    measured_attention = attention_loss(mask, mask)
    assert measured_attention.item() == 0.0, "attention_loss must be zero for equal maps"

    combined = total_c2_loss(
        noise,
        prediction,
        prediction,
        mask,
        mask,
        mask,
        alpha=0.1,
        lambda_obj=1.0,
        lambda_attn=0.1,
    )
    expected_total = measured_defect + measured_object + 0.1 * measured_attention
    assert torch.isclose(combined.total, expected_total), "total_c2_loss weight mismatch"


def _verify_object_mask_bounds(config: dict[str, object]) -> None:
    defect_mask = np.zeros((64, 64), dtype=np.uint8)
    defect_mask[20:30, 25:45] = 1
    object_mask = make_object_rectangle_mask(
        defect_mask,
        min_bbox_area_scale=float(config["object_mask_min_bbox_area_scale"]),
        max_bbox_area_scale=float(config["object_mask_max_bbox_area_scale"]),
        rng=random.Random(int(config["seed"])),
    )
    bbox_area = 10 * 20
    area_scale = float(object_mask.sum()) / bbox_area
    assert area_scale >= float(config["object_mask_min_bbox_area_scale"])
    assert area_scale <= float(config["object_mask_max_bbox_area_scale"])
    assert np.all(object_mask[defect_mask > 0] == 1), (
        "Object rectangle does not contain the full defect bbox"
    )


def _reload_artifacts(
    config: dict[str, object],
    lora_path: Path,
    token_path: Path,
) -> tuple[int, int]:
    base_model_id = str(config["base_model_id"])
    fresh_unet = UNet2DConditionModel.from_pretrained(
        base_model_id,
        subfolder="unet",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        use_safetensors=False,
    )
    fresh_unet.requires_grad_(False)
    inject_cross_attention_lora(
        fresh_unet,
        rank=int(config["rank"]),
        alpha=int(config["lora_alpha"]),
        target_modules=list(config["target_modules"]),
    )
    load_lora_checkpoint(fresh_unet, lora_path)
    loaded_lora = get_peft_model_state_dict(fresh_unet)
    saved_lora = load_file(str(lora_path), device="cpu")
    assert set(loaded_lora) == set(saved_lora), "Reloaded LoRA keys differ from checkpoint"
    assert loaded_lora, "Reloaded LoRA has no tensors"
    for key, value in loaded_lora.items():
        assert torch.isfinite(value).all().item(), f"Non-finite LoRA tensor: {key}"
        assert torch.equal(value.detach().cpu(), saved_lora[key]), (
            f"LoRA tensor changed during reload: {key}"
        )

    tokenizer = CLIPTokenizer.from_pretrained(
        base_model_id,
        subfolder="tokenizer",
        local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base_model_id,
        subfolder="text_encoder",
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    token_manager = LearnedTokenManager.load(tokenizer, text_encoder, token_path)
    saved_token = torch.load(token_path, map_location="cpu", weights_only=True)
    loaded_embedding = token_manager.embedding_layer.weight[token_manager.token_id]
    assert token_manager.token == str(config["learned_token"])
    assert torch.equal(
        loaded_embedding.detach().float().cpu(),
        saved_token["embedding"].float(),
    ), "Learned-token embedding changed during reload"
    token_manager.close()
    return len(loaded_lora), int(loaded_embedding.numel())


def main() -> None:
    started_at = time.perf_counter()
    config = load_training_config(DEFAULT_CONFIG_PATH)
    assert config["dataset_key"] == "mvtec_ad"
    assert config["class_name"] == "bottle"
    assert int(config["pair_budget"]) == 5
    assert int(config["rank"]) == 16
    assert int(config["lora_alpha"]) == 16
    assert list(config["target_modules"]) == ["to_q", "to_k", "to_v", "to_out.0"]
    assert float(config["learning_rate"]) == 1.0e-4
    assert int(config["smoke_steps"]) == 50
    assert int(config["max_steps"]) == 1000
    assert float(config["max_projected_pilot_minutes"]) == 120.0
    assert int(config["num_workers"]) == 0
    assert config["enable_masked_ti"] is False

    _verify_loss_equations()
    _verify_object_mask_bounds(config)

    verify_config_path, verify_config = _isolated_verify_config(config)

    result = train_lora_defect(
        verify_config_path,
        steps=int(verify_config["smoke_steps"]),
        enforce_smoke_gate=True,
        log_to_mlflow=True,
    )
    assert len(result.history) == 50, f"Expected 50 loss records, got {len(result.history)}"
    metric_names = ("loss_total", "loss_defect", "loss_object", "loss_attention")
    for record in result.history:
        for metric_name in metric_names:
            assert math.isfinite(record[metric_name]), (
                f"Non-finite {metric_name} at step {int(record['step'])}"
            )
    assert all(record["loss_attention"] > 0.0 for record in result.history), (
        "Attention-map hooks were skipped during the smoke run"
    )

    first_ten_mean = float(
        np.mean([record["loss_total"] for record in result.history[:10]])
    )
    last_ten_mean = float(
        np.mean([record["loss_total"] for record in result.history[-10:]])
    )
    assert last_ten_mean < first_ten_mean, (
        "Total loss did not decrease: "
        f"first-10 mean={first_ten_mean:.6f}, last-10 mean={last_ten_mean:.6f}"
    )

    expected_dir = (
        REPO_ROOT
        / "outputs"
        / "checkpoints"
        / "c2_phase2_verify"
        / str(config["dataset_key"])
        / str(config["class_name"])
    )
    assert result.lora_path == expected_dir / "lora.safetensors"
    assert result.token_path == expected_dir / "token.pt"
    assert result.lora_path.is_file() and result.lora_path.stat().st_size > 0
    assert result.token_path.is_file() and result.token_path.stat().st_size > 0

    gc.collect()
    torch.cuda.empty_cache()
    lora_tensor_count, token_dimensions = _reload_artifacts(
        verify_config,
        result.lora_path,
        result.token_path,
    )

    elapsed_seconds = time.perf_counter() - started_at
    print("Phase 2 verification passed")
    print(f"Pilot: {config['dataset_key']}/{config['class_name']} ({config['pair_budget']} pairs)")
    print(f"Steps: {len(result.history)}")
    print(f"First-10 total-loss mean: {first_ten_mean:.6f}")
    print(f"Last-10 total-loss mean: {last_ten_mean:.6f}")
    print(f"Training throughput: {result.steps_per_second:.4f} steps/s")
    print(f"Projected 1000-step wall time: {result.projected_1000_seconds / 60.0:.2f} minutes")
    print(f"Peak VRAM: {result.peak_vram_gib:.3f} GiB")
    print(f"Reloaded LoRA tensors: {lora_tensor_count}")
    print(f"Reloaded token dimensions: {token_dimensions}")
    print(f"LoRA checkpoint: {result.lora_path.relative_to(REPO_ROOT).as_posix()}")
    print(f"Token checkpoint: {result.token_path.relative_to(REPO_ROOT).as_posix()}")
    print(f"End-to-end verification time: {elapsed_seconds:.2f} s")


if __name__ == "__main__":
    main()
