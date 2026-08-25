"""Fallback-aware Qwen2.5-VL loading for Windows and Blackwell CUDA."""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Mapping

import torch
import yaml
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from src.c3_explanation.utils.logging_utils import get_logger

logger = get_logger(__name__)

_C3_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _C3_ROOT / "configs" / "qwen_vl_base.yaml"
_DTYPES = {"bfloat16": torch.bfloat16}
_QUANTIZATION_MODES = {"nf4", "int8", "bf16"}


def _repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value).replace("\\", "/"))
    return path if path.is_absolute() else _REPO_ROOT / path


def load_vlm_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load and validate the complete Phase 0 model/fallback configuration."""

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"C3 VLM config must be a mapping: {config_path}")
    required = {
        "model_id",
        "quantization",
        "compute_dtype",
        "device_map",
        "attention_implementation",
        "local_cache_dir",
        "local_files_only",
        "trust_remote_code",
        "seed",
        "generation",
        "fallback",
    }
    missing = required.difference(config)
    if missing:
        raise ValueError(f"C3 VLM config is missing keys: {sorted(missing)}")
    if str(config["compute_dtype"]) not in _DTYPES:
        raise ValueError("Phase 0 supports bfloat16 compute only")
    if str(config["quantization"]) != "nf4":
        raise ValueError("The preferred Phase 0 quantization must be nf4")
    if not isinstance(config["local_files_only"], bool):
        raise ValueError("local_files_only must be a boolean")
    if not isinstance(config["trust_remote_code"], bool):
        raise ValueError("trust_remote_code must be a boolean")

    generation = config["generation"]
    if not isinstance(generation, dict):
        raise ValueError("generation must be a mapping")
    if set(generation) != {"max_new_tokens", "temperature", "do_sample"}:
        raise ValueError(
            "generation requires exactly max_new_tokens, temperature, and do_sample"
        )
    if not 0 < int(generation["max_new_tokens"]) <= 16:
        raise ValueError("Phase 0 max_new_tokens must lie in [1,16]")
    if float(generation["temperature"]) != 0.0 or generation["do_sample"] is not False:
        raise ValueError("Phase 0 generation must use deterministic greedy decoding")

    fallback = config["fallback"]
    if not isinstance(fallback, dict):
        raise ValueError("fallback must be a mapping")
    if set(fallback) != {"quantization_fallback", "model_id_fallback"}:
        raise ValueError(
            "fallback requires exactly quantization_fallback and model_id_fallback"
        )
    modes = [str(mode) for mode in fallback["quantization_fallback"]]
    if modes != ["nf4", "int8", "bf16"] or any(
        mode not in _QUANTIZATION_MODES for mode in modes
    ):
        raise ValueError("Fallback order must be exactly nf4 -> int8 -> bf16")
    return config


def _quantization_config(mode: str, compute_dtype: torch.dtype) -> BitsAndBytesConfig | None:
    if mode == "nf4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    if mode == "int8":
        return BitsAndBytesConfig(load_in_8bit=True)
    if mode == "bf16":
        return None
    raise ValueError(f"Unsupported quantization mode: {mode}")


def _release_failed_attempt() -> None:
    """Release only this process's failed allocations; other GPU jobs are untouched."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_quantised_vlm(
    config: Mapping[str, Any],
) -> tuple[Qwen2_5_VLForConditionalGeneration, Any]:
    """Load Qwen2.5-VL through the configured NF4 -> INT8 -> bf16-3B chain.

    Every failure is recorded in ``model.c3_load_report`` and logged. Quantised
    attempts use the 7B model. The final bf16 attempt switches to the configured
    3B fallback and does not require bitsandbytes.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("Phase 0 requires a CUDA GPU")
    compute_dtype = _DTYPES[str(config["compute_dtype"])]
    fallback = config["fallback"]
    modes = [str(mode) for mode in fallback["quantization_fallback"]]
    cache_dir = _repo_path(str(config["local_cache_dir"]))
    cache_dir.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, str]] = []

    for mode in modes:
        model = None
        processor = None
        model_id = (
            str(fallback["model_id_fallback"])
            if mode == "bf16"
            else str(config["model_id"])
        )
        logger.info("Attempting C3 VLM load: model=%s precision=%s", model_id, mode)
        try:
            quantization_config = _quantization_config(mode, compute_dtype)
            load_kwargs: dict[str, Any] = {
                "cache_dir": cache_dir,
                "local_files_only": bool(config["local_files_only"]),
                "trust_remote_code": bool(config["trust_remote_code"]),
                "device_map": config["device_map"],
                "attn_implementation": str(config["attention_implementation"]),
                "dtype": compute_dtype,
            }
            if quantization_config is not None:
                load_kwargs["quantization_config"] = quantization_config
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                **load_kwargs,
            )
            processor = AutoProcessor.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                local_files_only=bool(config["local_files_only"]),
                trust_remote_code=bool(config["trust_remote_code"]),
            )
            model.eval()
            attempts.append(
                {"precision": mode, "model_id": model_id, "status": "succeeded"}
            )
            report = {
                "selected_precision": mode,
                "selected_model_id": model_id,
                "attempts": tuple(attempts),
            }
            model.c3_load_report = report  # type: ignore[attr-defined]
            processor.c3_load_report = report
            logger.info("C3 VLM load succeeded: model=%s precision=%s", model_id, mode)
            return model, processor
        except Exception as error:
            error_text = f"{type(error).__name__}: {error}"
            attempts.append(
                {
                    "precision": mode,
                    "model_id": model_id,
                    "status": "failed",
                    "error": error_text,
                }
            )
            logger.warning(
                "C3 VLM load failed: model=%s precision=%s error=%s",
                model_id,
                mode,
                error_text,
            )
            # A processor failure can occur after the model has reached CUDA.
            # Drop both local references before trying the next fallback so this
            # process does not retain VRAM alongside the active C2 training job.
            model = None
            processor = None
            _release_failed_attempt()

    rendered = "; ".join(
        f"{attempt['precision']}:{attempt['status']} ({attempt.get('error', '')})"
        for attempt in attempts
    )
    raise RuntimeError(f"All configured C3 VLM load attempts failed: {rendered}")
