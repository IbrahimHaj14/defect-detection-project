"""Phase 3 QLoRA training, preflight, and checkpoint validation entry point."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import shutil
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import mlflow
import torch
import yaml
from PIL import Image
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from qwen_vl_utils.vision_process import smart_resize
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, get_linear_schedule_with_warmup
from transformers.image_utils import SizeDict

from src.c3_explanation.model.load_vlm import (
    load_quantised_vlm,
    load_vlm_config,
)
from src.c3_explanation.model.report_prompt import (
    bridge_prompt_facts,
    build_report_messages,
    prompt_contract_hash,
    render_prefix_and_full,
    serialise_report,
)
from src.c3_explanation.utils.json_io import (
    load_json,
    save_json,
    validate_against_schema,
)
from src.c3_explanation.utils.mlflow_utils import start_c3_run
from src.c3_explanation.utils.seed import set_global_seed

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_QLORA_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "qlora_report.yaml"
)
_SUPPORTED_DATASETS = ("mvtec_ad", "ecf")
_FORBIDDEN_TRAINABLE_FRAGMENTS = (
    "visual",
    "merger",
    "embed_tokens",
    "lm_head",
    ".mlp.",
    ".norm",
)


def _repo_path(value: str | Path) -> Path:
    path = Path(str(value).replace("\\", "/"))
    return path if path.is_absolute() else _REPO_ROOT / path


def _load_yaml(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as input_file:
        value = yaml.safe_load(input_file)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a mapping: {input_path}")
    return value


def load_phase3_config(
    path: str | Path = DEFAULT_QLORA_CONFIG_PATH,
) -> dict[str, Any]:
    """Load and validate the approved Phase 3 experiment configuration."""

    config = _load_yaml(path)
    required_sections = {
        "version",
        "base_model",
        "lora",
        "prompt",
        "image",
        "sequence",
        "data",
        "training",
        "stages",
        "runtime_limits",
        "generation",
        "checkpoint",
        "mlflow",
    }
    missing = required_sections.difference(config)
    if missing:
        raise ValueError(f"Phase 3 config is missing sections: {sorted(missing)}")
    lora = config["lora"]
    if (
        int(lora["rank"]) != 32
        or int(lora["alpha"]) != 32
        or float(lora["lora_dropout"]) != 0.05
        or str(lora["bias"]) != "none"
        or str(lora["task_type"]) != "CAUSAL_LM"
        or list(lora["target_modules"]) != ["q_proj", "k_proj", "v_proj", "o_proj"]
        or lora["modules_to_save"] is not None
    ):
        raise ValueError("QLoRA settings differ from the approved Phase 3 contract")
    training = config["training"]
    if int(training["batch_size"]) != 1 or int(training["num_workers"]) != 0:
        raise ValueError("Phase 3 requires batch_size=1 and num_workers=0")
    if not bool(training["bf16"]) or bool(training["fp16"]) or bool(training["tf32"]):
        raise ValueError("Phase 3 precision must be bf16=true, fp16/tf32=false")
    if bool(config["sequence"]["truncation"]):
        raise ValueError("Phase 3 forbids sequence truncation")
    if str(config["sequence"]["padding"]) != "dynamic":
        raise ValueError("Phase 3 requires dynamic padding")
    if set(config["data"]["datasets"]) != set(_SUPPORTED_DATASETS):
        raise ValueError("Phase 3 requires separate mvtec_ad and ecf datasets")
    if str(config["mlflow"]["tracking_uri"]) != "file:./outputs/logs/mlflow":
        raise ValueError("Phase 3 must use the shared MLflow store")
    return config


@dataclass(frozen=True)
class CorpusRecord:
    dataset: str
    class_name: str
    defect_type: str
    uid: str
    image_path: Path
    grounding_path: Path
    report_path: Path
    grounding: Mapping[str, Any]
    report: Mapping[str, Any]
    image_width: int
    image_height: int
    vision_tokens: int

    @property
    def target_json(self) -> str:
        return serialise_report(self.report)


@dataclass
class PreflightSummary:
    dataset: str
    examples: int
    development_train_examples: int
    development_validation_examples: int
    minimum_sequence_tokens: int
    median_sequence_tokens: float
    p95_sequence_tokens: float
    p99_sequence_tokens: float
    maximum_sequence_tokens: int
    minimum_target_tokens: int
    maximum_target_tokens: int
    maximum_vision_tokens: int
    longest_examples: list[dict[str, Any]]
    membership_sha256: str


@dataclass
class TrainingSummary:
    dataset: str
    stage: str
    optimizer_updates: int
    micro_presentations: int
    losses: list[float]
    wall_time_seconds: float
    optimizer_updates_per_second: float
    peak_allocated_vram_bytes: int
    peak_reserved_vram_bytes: int
    physical_vram_bytes: int
    projected_full_run_hours: float
    trainable_parameters: int
    mlflow_run_id: str
    checkpoint_path: str | None = None
    generated_report: Mapping[str, Any] | None = None
    generated_raw_text: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _defect_vocabulary(dataset_config: Mapping[str, Any]) -> list[str]:
    vocabulary: set[str] = set()
    classes = dataset_config.get("classes")
    if not isinstance(classes, Mapping):
        raise ValueError("Dataset config classes must be a mapping")
    for class_config in classes.values():
        if not isinstance(class_config, Mapping):
            raise ValueError("Dataset class config must be a mapping")
        defect_types = class_config.get("defect_types")
        if not isinstance(defect_types, Mapping):
            raise ValueError("Dataset defect_types must be a mapping")
        vocabulary.update(str(item) for item in defect_types)
    return sorted(vocabulary)


def _vision_tokens_for_image(
    width: int,
    height: int,
    image_config: Mapping[str, Any],
) -> int:
    patch_size = int(image_config["patch_size"])
    merge_size = int(image_config["merge_size"])
    factor = patch_size * merge_size
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=factor,
        min_pixels=int(image_config["min_pixels"]),
        max_pixels=int(image_config["max_pixels"]),
    )
    return (resized_height // factor) * (resized_width // factor)


def load_training_records(
    dataset: str,
    config: Mapping[str, Any],
) -> tuple[list[CorpusRecord], list[str], str]:
    """Load only a dataset's Phase 2 training population and paired artifacts."""

    if dataset not in _SUPPORTED_DATASETS:
        raise ValueError(f"Unsupported C3 dataset: {dataset}")
    data_config = config["data"]
    dataset_spec = data_config["datasets"][dataset]
    corpus_dir = _repo_path(dataset_spec["corpus_dir"])
    membership_path = corpus_dir / str(data_config["membership_filename"])
    membership = load_json(membership_path)
    if membership.get("dataset_key") != dataset:
        raise ValueError(f"Membership dataset mismatch: {membership_path}")
    members = membership.get("train")
    if not isinstance(members, list):
        raise ValueError(f"Membership train population is missing: {membership_path}")
    expected = int(dataset_spec["expected_phase2_train_examples"])
    if len(members) != expected:
        raise ValueError(
            f"{dataset} Phase 2 training count changed: expected {expected}, got {len(members)}"
        )

    dataset_config = _load_yaml(_repo_path(dataset_spec["dataset_config"]))
    vocabulary = _defect_vocabulary(dataset_config)
    schema_path = _repo_path(data_config["report_schema_path"])
    records: list[CorpusRecord] = []
    seen_uids: set[str] = set()
    for member in members:
        uid = str(member["uid"])
        if uid in seen_uids:
            raise ValueError(f"Duplicate Phase 2 training UID: {uid}")
        seen_uids.add(uid)
        class_name = str(member["class"])
        defect_type = str(member["defect_type"])
        if defect_type not in vocabulary:
            raise ValueError(f"Cross-dataset or unknown defect type: {defect_type}")
        image_value = str(member["image_path"]).replace("\\", "/")
        mask_value = str(member["mask_path"]).replace("\\", "/")
        if "c2" in image_value.lower() or "synthetic" in image_value.lower():
            raise ValueError(f"C2/synthetic path entered Phase 3: {image_value}")
        image_path = _repo_path(image_value)
        grounding_path = corpus_dir / "train" / f"{uid}.grounding.json"
        report_path = corpus_dir / "train" / f"{uid}.report.json"
        for required_path in (image_path, grounding_path, report_path):
            if not required_path.is_file():
                raise FileNotFoundError(required_path)
        grounding = load_json(grounding_path)
        report = load_json(report_path)
        validate_against_schema(report, schema_path)
        if str(grounding.get("uid")) != uid or str(grounding.get("dataset")) != dataset:
            raise ValueError(f"Grounding identity mismatch for {uid}")
        if str(grounding.get("image_path")).replace("\\", "/") != image_value:
            raise ValueError(f"Grounding image path mismatch for {uid}")
        if str(grounding.get("mask_path")).replace("\\", "/") != mask_value:
            raise ValueError(f"Grounding mask path mismatch for {uid}")
        if report.get("defect_type") != defect_type:
            raise ValueError(f"Membership/report defect type mismatch for {uid}")
        action = report.get("recommended_action", {}).get("action")
        if action != "inspect":
            raise ValueError(f"Phase 2 action supervision is not inspect for {uid}")
        bridge_prompt_facts(grounding, report)
        with Image.open(image_path) as image:
            width, height = image.size
        vision_tokens = _vision_tokens_for_image(width, height, config["image"])
        records.append(
            CorpusRecord(
                dataset=dataset,
                class_name=class_name,
                defect_type=defect_type,
                uid=uid,
                image_path=image_path,
                grounding_path=grounding_path,
                report_path=report_path,
                grounding=grounding,
                report=report,
                image_width=width,
                image_height=height,
                vision_tokens=vision_tokens,
            )
        )
    records.sort(key=lambda item: (item.class_name, item.defect_type, item.uid))
    return records, vocabulary, _sha256_file(membership_path)


def _round_half_up(value: float) -> int:
    return math.floor(value + 0.5)


def development_split(
    records: Sequence[CorpusRecord],
    config: Mapping[str, Any],
) -> tuple[list[CorpusRecord], list[CorpusRecord]]:
    """Derive the approved runtime-only development split from Phase 2 train."""

    split_config = config["data"]["development_split"]
    fraction = float(split_config["fraction"])
    seed = int(split_config["seed"])
    strata: dict[tuple[str, str], list[CorpusRecord]] = {}
    for record in records:
        strata.setdefault((record.class_name, record.defect_type), []).append(record)
    import random

    generator = random.Random(seed)
    train: list[CorpusRecord] = []
    validation: list[CorpusRecord] = []
    for stratum in sorted(strata):
        candidates = sorted(strata[stratum], key=lambda item: item.uid)
        generator.shuffle(candidates)
        if len(candidates) == 1:
            validation_count = 0
        else:
            validation_count = _round_half_up(len(candidates) * fraction)
            validation_count = min(max(validation_count, 1), len(candidates) - 1)
        validation.extend(candidates[:validation_count])
        train.extend(candidates[validation_count:])
    train.sort(key=lambda item: (item.class_name, item.defect_type, item.uid))
    validation.sort(key=lambda item: (item.class_name, item.defect_type, item.uid))
    if set(item.uid for item in train).intersection(item.uid for item in validation):
        raise AssertionError("Development train/validation overlap")
    if {item.uid for item in train}.union(item.uid for item in validation) != {
        item.uid for item in records
    }:
        raise AssertionError("Development split does not partition Phase 2 train")
    return train, validation


def _build_messages_for_record(
    record: CorpusRecord,
    vocabulary: Sequence[str],
    config: Mapping[str, Any],
    *,
    include_target: bool,
    image: str | Path | Any | None = None,
) -> list[dict[str, Any]]:
    return build_report_messages(
        image=record.image_path if image is None else image,
        defect_vocabulary=vocabulary,
        bridge_facts=bridge_prompt_facts(record.grounding, record.report),
        prompt_config=config["prompt"],
        target_report=record.report if include_target else None,
    )


def _token_boundary(
    record: CorpusRecord,
    processor: Any,
    vocabulary: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[list[int], list[int], int, str]:
    prefix_messages = _build_messages_for_record(
        record, vocabulary, config, include_target=False
    )
    full_messages = _build_messages_for_record(
        record, vocabulary, config, include_target=True
    )
    target_json = record.target_json
    prefix_text, full_text, suffix_text = render_prefix_and_full(
        processor,
        messages_without_target=prefix_messages,
        messages_with_target=full_messages,
        target_json=target_json,
    )
    tokenizer = processor.tokenizer
    prefix_ids = tokenizer(prefix_text, add_special_tokens=False).input_ids
    full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
    if full_ids[: len(prefix_ids)] != prefix_ids:
        raise AssertionError(f"Tokenised assistant prefix mismatch for {record.uid}")
    suffix_ids = full_ids[len(prefix_ids) :]
    decoded = tokenizer.decode(
        suffix_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if decoded != suffix_text:
        raise AssertionError(f"Supervised assistant span decode mismatch for {record.uid}")
    image_pad_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    image_positions = [index for index, token in enumerate(full_ids) if token == image_pad_id]
    if len(image_positions) != 1:
        raise AssertionError(f"Expected one image placeholder for {record.uid}")

    forbidden_values = (
        record.uid,
        record.image_path.as_posix(),
        record.grounding_path.as_posix(),
        record.report_path.as_posix(),
        record.image_path.name,
    )
    if any(value and value in prefix_text for value in forbidden_values):
        raise AssertionError(f"Source metadata leaked into prompt for {record.uid}")
    forbidden_keys = ('"uid"', '"image_path"', '"mask_path"', '"class"', "class_name")
    if any(value in prefix_text for value in forbidden_keys):
        raise AssertionError(f"Bookkeeping/class field leaked into prompt for {record.uid}")
    return full_ids, suffix_ids, image_positions[0], full_text


def _expanded_masking_preflight(
    record: CorpusRecord,
    processor: Any,
    vocabulary: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[int, int]:
    full_ids, suffix_ids, image_position, _ = _token_boundary(
        record, processor, vocabulary, config
    )
    expanded_ids = (
        full_ids[:image_position]
        + [full_ids[image_position]] * record.vision_tokens
        + full_ids[image_position + 1 :]
    )
    target_start = len(expanded_ids) - len(suffix_ids)
    labels = [-100] * target_start + suffix_ids
    if len(labels) != len(expanded_ids):
        raise AssertionError(f"Label length mismatch for {record.uid}")
    supervised = [label for label in labels if label != -100]
    if not supervised:
        raise AssertionError(f"No supervised target tokens for {record.uid}")
    decoded = processor.tokenizer.decode(
        supervised,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    expected = f"{record.target_json}<|im_end|>\n"
    if decoded != expected:
        raise AssertionError(f"Expanded supervised span mismatch for {record.uid}")
    if len(expanded_ids) > int(config["sequence"]["max_sequence_length"]):
        raise ValueError(
            f"Complete example {record.uid} has {len(expanded_ids)} tokens, exceeding "
            f"the configured maximum {config['sequence']['max_sequence_length']}"
        )
    return len(expanded_ids), len(supervised)


def _percentile(values: Sequence[int], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def preflight_dataset(
    dataset: str,
    processor: Any,
    config: Mapping[str, Any],
) -> PreflightSummary:
    """Prove leakage, masking, sequence, and runtime-split contracts corpus-wide."""

    records, vocabulary, membership_hash = load_training_records(dataset, config)
    dev_train, dev_validation = development_split(records, config)
    sequence_lengths: list[int] = []
    target_lengths: list[int] = []
    details: list[tuple[int, CorpusRecord, int]] = []
    for index, record in enumerate(records, start=1):
        sequence_length, target_length = _expanded_masking_preflight(
            record, processor, vocabulary, config
        )
        sequence_lengths.append(sequence_length)
        target_lengths.append(target_length)
        details.append((sequence_length, record, target_length))
        if index % 250 == 0 or index == len(records):
            print(f"PREFLIGHT_PROGRESS dataset={dataset} examples={index}/{len(records)}")
    longest = []
    for sequence_length, record, target_length in sorted(
        details, key=lambda item: (item[0], item[1].uid), reverse=True
    )[:5]:
        longest.append(
            {
                "uid": record.uid,
                "sequence_tokens": sequence_length,
                "target_tokens": target_length,
                "vision_tokens": record.vision_tokens,
                "image_size": [record.image_width, record.image_height],
            }
        )
    return PreflightSummary(
        dataset=dataset,
        examples=len(records),
        development_train_examples=len(dev_train),
        development_validation_examples=len(dev_validation),
        minimum_sequence_tokens=min(sequence_lengths),
        median_sequence_tokens=statistics.median(sequence_lengths),
        p95_sequence_tokens=_percentile(sequence_lengths, 0.95),
        p99_sequence_tokens=_percentile(sequence_lengths, 0.99),
        maximum_sequence_tokens=max(sequence_lengths),
        minimum_target_tokens=min(target_lengths),
        maximum_target_tokens=max(target_lengths),
        maximum_vision_tokens=max(item.vision_tokens for item in records),
        longest_examples=longest,
        membership_sha256=membership_hash,
    )


def _configure_processor(processor: Any, config: Mapping[str, Any]) -> None:
    image_config = config["image"]
    processor.image_processor.size = SizeDict(
        shortest_edge=int(image_config["min_pixels"]),
        longest_edge=int(image_config["max_pixels"]),
    )
    if int(processor.image_processor.patch_size) != int(image_config["patch_size"]):
        raise ValueError("Processor patch size differs from Phase 3 config")
    if int(processor.image_processor.merge_size) != int(image_config["merge_size"]):
        raise ValueError("Processor merge size differs from Phase 3 config")


def encode_training_record(
    record: CorpusRecord,
    processor: Any,
    vocabulary: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Encode one image/report and mask every non-assistant-target position."""

    _, suffix_ids, _, full_text = _token_boundary(record, processor, vocabulary, config)
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
    encoded = processor(
        images=[image],
        text=[full_text],
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    input_ids = encoded["input_ids"]
    suffix_tensor = torch.tensor(suffix_ids, dtype=input_ids.dtype)
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise AssertionError(f"Unexpected processor batch shape for {record.uid}")
    if input_ids.shape[1] < len(suffix_ids):
        raise AssertionError(f"Encoded sequence is shorter than target for {record.uid}")
    if not torch.equal(input_ids[0, -len(suffix_ids) :].cpu(), suffix_tensor):
        raise AssertionError(f"Processed assistant suffix mismatch for {record.uid}")
    if input_ids.shape[1] > int(config["sequence"]["max_sequence_length"]):
        raise ValueError(f"Encoded example exceeds sequence limit: {record.uid}")
    labels = torch.full_like(input_ids, -100)
    labels[0, -len(suffix_ids) :] = input_ids[0, -len(suffix_ids) :]
    if int((labels != -100).sum()) != len(suffix_ids):
        raise AssertionError(f"Incorrect supervised-token count for {record.uid}")
    encoded["labels"] = labels
    return {key: value for key, value in encoded.items() if isinstance(value, torch.Tensor)}


class EncodedCorpusDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: Sequence[CorpusRecord],
        processor: Any,
        vocabulary: Sequence[str],
        config: Mapping[str, Any],
    ) -> None:
        self.records = list(records)
        self.processor = processor
        self.vocabulary = list(vocabulary)
        self.config = config

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return encode_training_record(
            self.records[index], self.processor, self.vocabulary, self.config
        )


def _single_item_collate(
    batch: Sequence[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    if len(batch) != 1:
        raise AssertionError("Phase 3 collator requires batch_size=1")
    return batch[0]


def build_training_dataloader(
    records: Sequence[CorpusRecord],
    processor: Any,
    vocabulary: Sequence[str],
    config: Mapping[str, Any],
    *,
    epoch: int,
    shuffle: bool,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build the Windows-safe seeded natural-frequency DataLoader."""

    training = config["training"]
    generator = torch.Generator()
    generator.manual_seed(int(training["data_seed"]) + epoch)
    loader = DataLoader(
        EncodedCorpusDataset(records, processor, vocabulary, config),
        batch_size=int(training["batch_size"]),
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        collate_fn=_single_item_collate,
    )
    if loader.num_workers != 0:
        raise AssertionError("Every C3 DataLoader must use num_workers=0")
    return loader


def _batch_stream(
    records: Sequence[CorpusRecord],
    processor: Any,
    vocabulary: Sequence[str],
    config: Mapping[str, Any],
    *,
    presentations: int,
    shuffle: bool,
) -> Iterator[dict[str, torch.Tensor]]:
    yielded = 0
    epoch = 0
    while yielded < presentations:
        loader = build_training_dataloader(
            records,
            processor,
            vocabulary,
            config,
            epoch=epoch,
            shuffle=shuffle,
        )
        for batch in loader:
            yield batch
            yielded += 1
            if yielded >= presentations:
                return
        epoch += 1


def select_smoke_records(
    records: Sequence[CorpusRecord],
    count: int,
) -> list[CorpusRecord]:
    """Select deterministic vision-token quantiles, including the maximum."""

    if count > len(records):
        raise ValueError("Smoke selection exceeds available development training records")
    ordered = sorted(records, key=lambda item: (item.vision_tokens, item.uid))
    indices = [round(index * (len(ordered) - 1) / (count - 1)) for index in range(count)]
    if len(set(indices)) != count:
        raise AssertionError("Smoke quantile selection did not produce unique examples")
    selected = [ordered[index] for index in indices]
    if selected[-1].vision_tokens != max(item.vision_tokens for item in records):
        raise AssertionError("Smoke selection omitted the maximum vision-token example")
    return selected


def _strict_load_base(config: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    base_config = config["base_model"]
    phase0 = load_vlm_config(_repo_path(base_config["config_path"]))
    runtime = copy.deepcopy(phase0)
    runtime["local_files_only"] = bool(base_config["local_files_only"])
    runtime["fallback"]["quantization_fallback"] = [
        str(base_config["required_precision"])
    ]
    model, processor = load_quantised_vlm(runtime)
    report = dict(model.c3_load_report)
    if report.get("selected_model_id") != str(base_config["required_model_id"]):
        raise RuntimeError(f"Canonical Phase 3 loaded the wrong model: {report}")
    if report.get("selected_precision") != str(base_config["required_precision"]):
        raise RuntimeError(f"Canonical Phase 3 loaded the wrong precision: {report}")
    if not bool(getattr(model, "is_loaded_in_4bit", False)):
        raise RuntimeError("Canonical Phase 3 base is not actually loaded in 4-bit")
    device_map = getattr(model, "hf_device_map", {})
    if bool(base_config["reject_cpu_or_disk_offload"]):
        invalid = {
            name: device
            for name, device in device_map.items()
            if str(device).lower() in {"cpu", "disk"}
        }
        if invalid:
            raise RuntimeError(f"CPU/disk offload is forbidden for Phase 3: {invalid}")
    _configure_processor(processor, config)
    return model, processor, report


def attach_qlora(model: Any, config: Mapping[str, Any]) -> tuple[Any, list[str], int]:
    """Attach adapters and assert the complete trainable-parameter policy."""

    training = config["training"]
    lora = config["lora"]
    if bool(training["gradient_checkpointing"]):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=True,
            gradient_checkpointing_kwargs={
                "use_reentrant": bool(training["gradient_checkpointing_use_reentrant"])
            },
        )
    else:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model.config.use_cache = bool(training["use_cache"])
    adapter_config = LoraConfig(
        r=int(lora["rank"]),
        lora_alpha=int(lora["alpha"]),
        lora_dropout=float(lora["lora_dropout"]),
        bias=str(lora["bias"]),
        task_type=TaskType[str(lora["task_type"])],
        target_modules=list(lora["target_modules"]),
        modules_to_save=lora["modules_to_save"],
    )
    model = get_peft_model(model, adapter_config)
    trainable_names: list[str] = []
    trainable_count = 0
    target_modules = tuple(str(item) for item in lora["target_modules"])
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_names.append(name)
        trainable_count += parameter.numel()
        lowered = name.lower()
        if ".lora_a." not in lowered and ".lora_b." not in lowered:
            raise RuntimeError(f"Unauthorised non-LoRA trainable parameter: {name}")
        if not any(f".{module}." in name for module in target_modules):
            raise RuntimeError(f"LoRA parameter targets an unauthorised module: {name}")
        if ".model.layers." not in name:
            raise RuntimeError(f"LoRA parameter is outside language-model layers: {name}")
        if any(fragment in lowered for fragment in _FORBIDDEN_TRAINABLE_FRAGMENTS):
            raise RuntimeError(f"Forbidden component became trainable: {name}")
    if not trainable_names:
        raise RuntimeError("No QLoRA parameters became trainable")
    expected = int(lora["expected_trainable_parameters"])
    if trainable_count != expected:
        raise RuntimeError(
            f"Trainable parameter count changed: expected {expected}, got {trainable_count}"
        )
    print(f"TRAINABLE_PARAMETERS count={trainable_count} tensors={len(trainable_names)}")
    for name in trainable_names:
        print(f"TRAINABLE_NAME {name}")
    model.train()
    return model, trainable_names, trainable_count


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=False) for name, tensor in batch.items()}


def _assert_lora_gradients(model: Any) -> None:
    missing: list[str] = []
    non_finite: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            missing.append(name)
        elif not torch.isfinite(parameter.grad).all():
            non_finite.append(name)
    if missing:
        raise RuntimeError(f"Trainable LoRA gradients are missing: {missing[:5]}")
    if non_finite:
        raise RuntimeError(f"Non-finite LoRA gradients: {non_finite[:5]}")


def _mlflow_params(
    dataset: str,
    stage: str,
    config: Mapping[str, Any],
    load_report: Mapping[str, Any],
    trainable_count: int,
    optimizer_updates: int,
    micro_presentations: int,
) -> dict[str, Any]:
    training = config["training"]
    return {
        "phase": 3,
        "run_stage": stage,
        "dataset": dataset,
        "model_id": load_report["selected_model_id"],
        "requested_precision": config["base_model"]["required_precision"],
        "selected_precision": load_report["selected_precision"],
        "rank": config["lora"]["rank"],
        "alpha": config["lora"]["alpha"],
        "lora_dropout": config["lora"]["lora_dropout"],
        "target_modules": config["lora"]["target_modules"],
        "trainable_parameter_count": trainable_count,
        "learning_rate": training["learning_rate"],
        "optimizer": training["optimizer"],
        "scheduler": training["scheduler"],
        "warmup_ratio": training["warmup_ratio"],
        "batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "effective_batch_size": int(training["batch_size"])
        * int(training["gradient_accumulation_steps"]),
        "optimizer_updates": optimizer_updates,
        "micro_presentations": micro_presentations,
        "min_pixels": config["image"]["min_pixels"],
        "max_pixels": config["image"]["max_pixels"],
        "max_sequence_length": config["sequence"]["max_sequence_length"],
        "seed": training["seed"],
        "data_seed": training["data_seed"],
        "prompt_version": config["prompt"]["version"],
    }


def _save_recovery_checkpoint(
    model: Any,
    processor: Any,
    run_dir: Path,
    update: int,
    retained: list[Path],
    save_limit: int,
) -> None:
    destination = run_dir / f"step-{update:06d}"
    model.save_pretrained(destination, safe_serialization=True)
    processor.save_pretrained(destination)
    retained.append(destination)
    while len(retained) > save_limit:
        obsolete = retained.pop(0)
        resolved = obsolete.resolve()
        if run_dir.resolve() not in resolved.parents:
            raise RuntimeError(f"Refusing to remove checkpoint outside recovery run: {resolved}")
        shutil.rmtree(resolved)


def _training_loop(
    *,
    model: Any,
    processor: Any,
    records: Sequence[CorpusRecord],
    vocabulary: Sequence[str],
    config: Mapping[str, Any],
    optimizer_updates: int,
    micro_presentations: int,
    shuffle: bool,
    recovery_interval: int | None,
    recovery_dir: Path | None,
) -> tuple[list[float], float]:
    training = config["training"]
    if str(training["optimizer"]) != "adamw_torch":
        raise ValueError("Only the approved adamw_torch optimizer is supported")
    if str(training["scheduler"]) != "linear":
        raise ValueError("Only the approved linear scheduler is supported")
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        betas=(float(training["adam_beta1"]), float(training["adam_beta2"])),
        eps=float(training["adam_epsilon"]),
        weight_decay=float(training["weight_decay"]),
    )
    warmup_steps = math.ceil(float(training["warmup_ratio"]) * optimizer_updates)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=optimizer_updates,
    )
    stream = _batch_stream(
        records,
        processor,
        vocabulary,
        config,
        presentations=micro_presentations,
        shuffle=shuffle,
    )
    device = next(parameter for parameter in model.parameters() if parameter.device.type == "cuda").device
    accumulation = int(training["gradient_accumulation_steps"])
    remaining = micro_presentations
    losses: list[float] = []
    retained: list[Path] = []
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    for update in range(1, optimizer_updates + 1):
        group_size = min(accumulation, remaining)
        if group_size <= 0:
            raise AssertionError("Optimizer update accounting exceeded presentations")
        micro_losses: list[float] = []
        for _ in range(group_size):
            batch = _move_batch(next(stream), device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                outputs = model(**batch)
                loss = outputs.loss.float()
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at optimizer update {update}")
            micro_losses.append(float(loss.detach().cpu()))
            (loss / group_size).backward()
        _assert_lora_gradients(model)
        torch.nn.utils.clip_grad_norm_(trainable, float(training["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        remaining -= group_size
        update_loss = statistics.fmean(micro_losses)
        losses.append(update_loss)
        elapsed = time.perf_counter() - started
        print(
            f"TRAIN_STEP update={update}/{optimizer_updates} loss={update_loss:.8f} "
            f"updates_per_second={update / elapsed:.6f}"
        )
        mlflow.log_metric("loss", update_loss, step=update)
        mlflow.log_metric("learning_rate", scheduler.get_last_lr()[0], step=update)
        if (
            recovery_interval
            and recovery_dir is not None
            and update < optimizer_updates
            and update % recovery_interval == 0
        ):
            _save_recovery_checkpoint(
                model,
                processor,
                recovery_dir,
                update,
                retained,
                int(training["save_total_limit"]),
            )
    if remaining != 0 or len(losses) != optimizer_updates:
        raise AssertionError(
            f"Training accounting mismatch: remaining={remaining}, losses={len(losses)}"
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return losses, time.perf_counter() - started


def _package_versions() -> dict[str, str]:
    packages = (
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "qwen-vl-utils",
        "Pillow",
        "mlflow",
    )
    return {name: importlib.metadata.version(name) for name in packages}


def _cached_revision(model: Any) -> str:
    revision = getattr(model.config, "_commit_hash", None)
    return str(revision) if revision else "unknown"


def _checkpoint_manifest(
    *,
    dataset: str,
    config: Mapping[str, Any],
    membership_hash: str,
    load_report: Mapping[str, Any],
    cached_revision: str,
    trainable_count: int,
    training_summary: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "phase": 3,
        "dataset": dataset,
        "base_model": load_report["selected_model_id"],
        "cached_revision": cached_revision,
        "actual_precision": load_report["selected_precision"],
        "membership_sha256": membership_hash,
        "trainable_parameter_count": trainable_count,
        "prompt_version": config["prompt"]["version"],
        "prompt_serialization_sha256": prompt_contract_hash(config["prompt"]),
        "training_config": {
            "lora": copy.deepcopy(config["lora"]),
            "image": copy.deepcopy(config["image"]),
            "sequence": copy.deepcopy(config["sequence"]),
            "training": copy.deepcopy(config["training"]),
            "full_stage": copy.deepcopy(config["stages"]["full"]),
            "dataset": copy.deepcopy(config["data"]["datasets"][dataset]),
        },
        "package_versions": _package_versions(),
        "seed": int(config["training"]["seed"]),
        "result": dict(training_summary),
    }


def _clear_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate_adapter_files(path: Path, config: Mapping[str, Any]) -> None:
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "preprocessor_config.json",
        "tokenizer_config.json",
    }
    names = {item.name for item in path.iterdir() if item.is_file()}
    missing = required.difference(names)
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing reproducibility files: {sorted(missing)}")
    forbidden = {"model.safetensors", "pytorch_model.bin"}.intersection(names)
    if forbidden:
        raise RuntimeError(f"Full base-model weights were saved unexpectedly: {forbidden}")
    adapter_config = load_json(path / "adapter_config.json")
    expected_targets = sorted(str(item) for item in config["lora"]["target_modules"])
    actual_targets = sorted(str(item) for item in adapter_config["target_modules"])
    if actual_targets != expected_targets:
        raise RuntimeError("Saved adapter target modules differ from Phase 3 contract")


def validate_checkpoint_artifacts(path: Path, config: Mapping[str, Any]) -> None:
    """Validate saved adapter/processor/manifest contents without loading the VLM."""

    _validate_adapter_files(path, config)
    manifest_path = path / str(config["checkpoint"]["manifest_filename"])
    manifest = load_json(manifest_path)
    if int(manifest.get("phase", -1)) != 3:
        raise ValueError(f"Checkpoint manifest is not Phase 3: {manifest_path}")
    if manifest.get("actual_precision") != config["base_model"]["required_precision"]:
        raise ValueError(f"Checkpoint precision is not canonical NF4: {manifest_path}")
    if manifest.get("base_model") != config["base_model"]["required_model_id"]:
        raise ValueError(f"Checkpoint base model is not canonical 7B: {manifest_path}")


def validate_staged_checkpoint(path: Path, config: Mapping[str, Any]) -> None:
    """Reload a staged adapter/processor against a fresh strict NF4 base."""

    _validate_adapter_files(path, config)
    base, _, _ = _strict_load_base(config)
    reloaded = PeftModel.from_pretrained(base, path, is_trainable=False)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    _configure_processor(processor, config)
    if not isinstance(reloaded, PeftModel):
        raise RuntimeError("Staged adapter did not reload as a PEFT model")
    if any(parameter.requires_grad for parameter in reloaded.parameters()):
        raise RuntimeError("Inference reload unexpectedly has trainable parameters")
    del reloaded, processor, base
    _clear_cuda_cache()


def _save_canonical_checkpoint(
    *,
    dataset: str,
    model: Any,
    processor: Any,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Path:
    canonical = _repo_path(config["data"]["datasets"][dataset]["checkpoint_dir"])
    staging = canonical.with_name(canonical.name + str(config["checkpoint"]["staging_suffix"]))
    if canonical.exists():
        raise FileExistsError(f"Refusing to overwrite canonical checkpoint: {canonical}")
    if staging.exists():
        raise FileExistsError(f"Refusing to overwrite stale staging checkpoint: {staging}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(staging, safe_serialization=True)
    processor.save_pretrained(staging)
    save_json(staging / str(config["checkpoint"]["manifest_filename"]), manifest)
    _validate_adapter_files(staging, config)
    return staging


def _strict_generate(
    *,
    model: Any,
    processor: Any,
    record: CorpusRecord,
    vocabulary: Sequence[str],
    config: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]]:
    messages = _build_messages_for_record(
        record, vocabulary, config, include_target=False
    )
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    with Image.open(record.image_path) as source:
        image = source.convert("RGB")
    inputs = processor(
        images=[image],
        text=[prompt],
        return_tensors="pt",
        padding=False,
        truncation=False,
    )
    device = next(parameter for parameter in model.parameters() if parameter.device.type == "cuda").device
    inputs = _move_batch(inputs, device)
    generation = config["generation"]
    model.eval()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(generation["max_new_tokens"]),
            do_sample=bool(generation["do_sample"]),
            temperature=float(generation["temperature"]),
            num_beams=int(generation["num_beams"]),
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    new_tokens = generated[:, inputs["input_ids"].shape[1] :]
    raw = processor.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    parsed = json.loads(raw)
    validate_against_schema(parsed, _repo_path(config["data"]["report_schema_path"]))
    return raw, parsed


def _temporary_checkpoint_dir(dataset: str, stage: str, config: Mapping[str, Any]) -> Path:
    root = _repo_path(config["checkpoint"]["temporary_root"])
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{dataset}-{stage}-", dir=root))


def run_training_stage(
    dataset: str,
    stage: str,
    config: Mapping[str, Any],
) -> TrainingSummary:
    """Run one approved Phase 3 stage without crossing into Phase 4."""

    if stage not in {"technical_smoke", "acceptance", "full"}:
        raise ValueError(f"Unsupported Phase 3 stage: {stage}")
    if stage == "acceptance" and dataset != str(config["stages"]["acceptance"]["dataset"]):
        raise ValueError("Phase 3 acceptance is defined on MVTec only")
    set_global_seed(int(config["training"]["seed"]))
    torch.backends.cuda.matmul.allow_tf32 = bool(config["training"]["tf32"])
    torch.backends.cudnn.allow_tf32 = bool(config["training"]["tf32"])
    records, vocabulary, membership_hash = load_training_records(dataset, config)
    dev_train, dev_validation = development_split(records, config)
    dataset_spec = config["data"]["datasets"][dataset]
    accumulation = int(config["training"]["gradient_accumulation_steps"])

    if stage == "technical_smoke":
        optimizer_updates = int(config["stages"][stage]["optimizer_updates"])
        micro_presentations = int(config["stages"][stage]["micro_presentations"])
        stage_records = select_smoke_records(dev_train, micro_presentations)
        shuffle = False
        recovery_interval = None
    elif stage == "acceptance":
        optimizer_updates = int(config["stages"][stage]["optimizer_updates"])
        micro_presentations = int(config["stages"][stage]["micro_presentations"])
        stage_records = dev_train
        shuffle = True
        recovery_interval = None
    else:
        optimizer_updates = int(dataset_spec["full_max_steps"])
        micro_presentations = int(dataset_spec["full_micro_presentations"])
        intended = int(config["stages"]["full"]["passes"]) * len(records)
        if micro_presentations != intended:
            raise RuntimeError(
                f"{dataset} full exposure mismatch: config={micro_presentations}, intended={intended}"
            )
        if optimizer_updates != math.ceil(micro_presentations / accumulation):
            raise RuntimeError(
                f"{dataset} update accounting mismatch for continuous accumulation"
            )
        stage_records = records
        shuffle = True
        recovery_interval = int(dataset_spec["recovery_checkpoint_steps"])

    model, processor, load_report = _strict_load_base(config)
    cached_revision = _cached_revision(model)
    model, _, trainable_count = attach_qlora(model, config)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    temporary_run_dir: Path | None = None
    if stage == "full":
        temporary_run_dir = _temporary_checkpoint_dir(dataset, "recovery", config)
    params = _mlflow_params(
        dataset,
        stage,
        config,
        load_report,
        trainable_count,
        optimizer_updates,
        micro_presentations,
    )
    experiment = str(dataset_spec["mlflow_experiment"])
    run_name = str(config["mlflow"]["run_stage_names"][stage])
    with start_c3_run(experiment, run_name, params) as run:
        losses, wall_time = _training_loop(
            model=model,
            processor=processor,
            records=stage_records,
            vocabulary=vocabulary,
            config=config,
            optimizer_updates=optimizer_updates,
            micro_presentations=micro_presentations,
            shuffle=shuffle,
            recovery_interval=recovery_interval,
            recovery_dir=temporary_run_dir,
        )
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
        physical_vram = int(torch.cuda.get_device_properties(0).total_memory)
        updates_per_second = optimizer_updates / wall_time
        projected_hours = (
            int(dataset_spec["full_max_steps"]) / updates_per_second / 3600.0
        )
        mlflow.log_metrics(
            {
                "wall_time_seconds": wall_time,
                "optimizer_updates_per_second": updates_per_second,
                "peak_allocated_vram_bytes": peak_allocated,
                "peak_reserved_vram_bytes": peak_reserved,
                "projected_full_run_hours": projected_hours,
            }
        )
        run_id = run.info.run_id

    summary = TrainingSummary(
        dataset=dataset,
        stage=stage,
        optimizer_updates=optimizer_updates,
        micro_presentations=micro_presentations,
        losses=losses,
        wall_time_seconds=wall_time,
        optimizer_updates_per_second=updates_per_second,
        peak_allocated_vram_bytes=peak_allocated,
        peak_reserved_vram_bytes=peak_reserved,
        physical_vram_bytes=physical_vram,
        projected_full_run_hours=projected_hours,
        trainable_parameters=trainable_count,
        mlflow_run_id=run_id,
    )
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError("Training recorded a non-finite loss")
    if peak_reserved / physical_vram >= float(
        config["runtime_limits"]["maximum_reserved_vram_fraction"]
    ):
        raise RuntimeError("Reserved VRAM reached the approved 90% abort threshold")
    if stage == "technical_smoke" and projected_hours > float(
        config["runtime_limits"]["maximum_projected_hours_per_adapter"]
    ):
        raise RuntimeError(
            f"Projected {dataset} full run is {projected_hours:.2f}h, over the 12h limit"
        )

    if stage == "acceptance":
        acceptance = config["stages"]["acceptance"]
        minimum = int(acceptance["minimum_loss_values"])
        window = int(acceptance["loss_window"])
        if len(losses) < minimum:
            raise RuntimeError(f"Acceptance recorded only {len(losses)} losses")
        first_mean = statistics.fmean(losses[:window])
        final_mean = statistics.fmean(losses[-window:])
        print(
            f"ACCEPTANCE_LOSS first_{window}_mean={first_mean:.8f} "
            f"last_{window}_mean={final_mean:.8f}"
        )
        if not final_mean < first_mean:
            raise RuntimeError("Acceptance loss did not decrease: last-5 mean >= first-5 mean")
        temporary = _temporary_checkpoint_dir(dataset, "acceptance", config)
        model.save_pretrained(temporary, safe_serialization=True)
        processor.save_pretrained(temporary)
        _validate_adapter_files(temporary, config)
        summary.checkpoint_path = temporary.relative_to(_REPO_ROOT).as_posix()
        del model, processor
        _clear_cuda_cache()
        base, _, _ = _strict_load_base(config)
        reloaded = PeftModel.from_pretrained(base, temporary, is_trainable=False)
        reloaded_processor = AutoProcessor.from_pretrained(temporary, local_files_only=True)
        _configure_processor(reloaded_processor, config)
        if not dev_validation:
            raise RuntimeError("MVTec development validation subset is empty")
        validation_record = dev_validation[0]
        raw, parsed = _strict_generate(
            model=reloaded,
            processor=reloaded_processor,
            record=validation_record,
            vocabulary=vocabulary,
            config=config,
        )
        summary.generated_raw_text = raw
        summary.generated_report = parsed
        save_json(temporary / "acceptance_summary.json", asdict(summary))
        del reloaded, reloaded_processor, base
        _clear_cuda_cache()
    elif stage == "full":
        result_for_manifest = asdict(summary)
        manifest = _checkpoint_manifest(
            dataset=dataset,
            config=config,
            membership_hash=membership_hash,
            load_report=load_report,
            cached_revision=cached_revision,
            trainable_count=trainable_count,
            training_summary=result_for_manifest,
        )
        staging = _save_canonical_checkpoint(
            dataset=dataset,
            model=model,
            processor=processor,
            config=config,
            manifest=manifest,
        )
        del model, processor
        _clear_cuda_cache()
        validate_staged_checkpoint(staging, config)
        canonical = _repo_path(dataset_spec["checkpoint_dir"])
        staging.rename(canonical)
        summary.checkpoint_path = canonical.relative_to(_REPO_ROOT).as_posix()
    else:
        del model, processor
        _clear_cuda_cache()
    print("PHASE3_STAGE_RESULT " + json.dumps(asdict(summary), sort_keys=True, default=str))
    return summary


def run_cpu_preflight(config: Mapping[str, Any]) -> dict[str, PreflightSummary]:
    """Run both complete-corpus preflights without loading the VLM."""

    base_config = load_vlm_config(_repo_path(config["base_model"]["config_path"]))
    processor = AutoProcessor.from_pretrained(
        str(config["base_model"]["required_model_id"]),
        cache_dir=_repo_path(base_config["local_cache_dir"]),
        local_files_only=True,
        trust_remote_code=bool(base_config["trust_remote_code"]),
    )
    _configure_processor(processor, config)
    summaries = {
        dataset: preflight_dataset(dataset, processor, config)
        for dataset in _SUPPORTED_DATASETS
    }
    for summary in summaries.values():
        print("PREFLIGHT_RESULT " + json.dumps(asdict(summary), sort_keys=True))
    return summaries


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_QLORA_CONFIG_PATH)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dataset", choices=_SUPPORTED_DATASETS)
    parser.add_argument(
        "--stage", choices=("technical_smoke", "acceptance", "full")
    )
    parser.add_argument("--validate-checkpoint", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    config = load_phase3_config(args.config)
    if args.preflight:
        run_cpu_preflight(config)
        return 0
    if args.validate_checkpoint is not None:
        validate_staged_checkpoint(args.validate_checkpoint, config)
        print(f"CHECKPOINT_VALID path={args.validate_checkpoint}")
        return 0
    if args.dataset and args.stage:
        run_training_stage(args.dataset, args.stage, config)
        return 0
    raise SystemExit("Choose --preflight, --validate-checkpoint, or --dataset with --stage")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_QLORA_CONFIG_PATH",
    "CorpusRecord",
    "PreflightSummary",
    "TrainingSummary",
    "attach_qlora",
    "build_training_dataloader",
    "development_split",
    "encode_training_record",
    "load_phase3_config",
    "load_training_records",
    "preflight_dataset",
    "run_cpu_preflight",
    "run_training_stage",
    "select_smoke_records",
    "validate_checkpoint_artifacts",
    "validate_staged_checkpoint",
]
