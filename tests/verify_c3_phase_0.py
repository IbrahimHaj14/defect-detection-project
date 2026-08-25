"""C3 Phase 0 acceptance: fallback-aware Qwen2.5-VL generation."""

from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.c3_explanation.model.load_vlm import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    load_quantised_vlm,
    load_vlm_config,
)
from src.c3_explanation.utils.json_io import (  # noqa: E402,F401
    load_json,
    save_json,
    validate_against_schema,
)
from src.c3_explanation.utils.logging_utils import get_logger  # noqa: E402,F401
from src.c3_explanation.utils.mlflow_utils import start_c3_run  # noqa: E402,F401
from src.c3_explanation.utils.seed import set_global_seed  # noqa: E402


def _generate_one_sentence(model: object, processor: object, config: dict[str, object]) -> str:
    image = Image.fromarray(np.full((448, 448, 3), 127, dtype=np.uint8), mode="RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)
    generation = config["generation"]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=bool(generation["do_sample"]),
        )
    trimmed_ids = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs.input_ids, output_ids, strict=True)
    ]
    return processor.batch_decode(
        trimmed_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main() -> None:
    started_at = time.perf_counter()
    config = load_vlm_config(DEFAULT_CONFIG_PATH)
    assert config["generation"]["do_sample"] is False
    assert float(config["generation"]["temperature"]) == 0.0
    assert int(config["generation"]["max_new_tokens"]) <= 16
    assert config["fallback"]["quantization_fallback"] == ["nf4", "int8", "bf16"]
    assert torch.cuda.is_available(), "C3 Phase 0 requires CUDA"

    set_global_seed(int(config["seed"]))
    torch.cuda.reset_peak_memory_stats()
    model, processor = load_quantised_vlm(config)
    assert model is not None and processor is not None
    report = getattr(model, "c3_load_report", None)
    assert isinstance(report, dict), "Loader did not attach its fallback report"
    assert report["selected_precision"] in {"nf4", "int8", "bf16"}

    generated_text = _generate_one_sentence(model, processor, config)
    assert generated_text, "Qwen2.5-VL returned empty text"
    peak_vram_gib = torch.cuda.max_memory_allocated() / (1024**3)
    elapsed_seconds = time.perf_counter() - started_at
    attempts = report["attempts"]
    fallback_path = " -> ".join(
        f"{attempt['precision']}:{attempt['status']}" for attempt in attempts
    )
    diagnostics = [
        f"{attempt['precision']} {attempt['model_id']}: {attempt.get('error', 'none')}"
        for attempt in attempts
        if attempt["status"] == "failed"
    ]

    print("C3 Phase 0 verification passed")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Fallback path used: {fallback_path}")
    print(f"Selected model: {report['selected_model_id']}")
    print(f"Selected precision: {report['selected_precision']}")
    print(f"Peak VRAM: {peak_vram_gib:.3f} GiB")
    print(f"Elapsed time: {elapsed_seconds:.2f} s")
    print(f"Generated text: {generated_text}")
    print("Warnings/errors: " + (" | ".join(diagnostics) if diagnostics else "none"))

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
