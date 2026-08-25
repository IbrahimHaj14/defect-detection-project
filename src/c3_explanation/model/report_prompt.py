"""Shared Phase 3/4 report-prompt and deterministic serialization contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

_BRIDGE_INPUT_KEYS = (
    "defect_present",
    "bounding_box",
    "centroid",
    "region",
    "affected_area_pct",
    "severity_level",
)


def compact_json(value: Any) -> str:
    """Serialize one JSON value using the approved deterministic representation."""

    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


def serialise_report(report: Mapping[str, Any]) -> str:
    """Serialize a complete target report without fencing or trailing prose."""

    return compact_json(dict(report))


def bridge_prompt_facts(
    grounding: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    """Return only Bridge evidence authorised as textual Brain input."""

    location = report.get("location")
    severity = report.get("severity")
    if not isinstance(location, Mapping) or not isinstance(severity, Mapping):
        raise ValueError("Report location and severity must be mappings")
    facts: dict[str, Any] = {
        "defect_present": report.get("defect_present"),
        "bounding_box": grounding.get("bounding_box"),
        "centroid": grounding.get("centroid"),
        "region": grounding.get("region"),
        "affected_area_pct": grounding.get("area_pct"),
        "severity_level": grounding.get("severity_level"),
    }
    if tuple(facts) != _BRIDGE_INPUT_KEYS:
        raise AssertionError("Bridge prompt-fact contract changed unexpectedly")
    if facts["bounding_box"] != location.get("bounding_box"):
        raise ValueError("Grounding/report bounding boxes disagree")
    if facts["centroid"] != location.get("centroid"):
        raise ValueError("Grounding/report centroids disagree")
    if facts["region"] != location.get("region"):
        raise ValueError("Grounding/report regions disagree")
    if facts["affected_area_pct"] != severity.get("affected_area_pct"):
        raise ValueError("Grounding/report affected areas disagree")
    if facts["severity_level"] != severity.get("level"):
        raise ValueError("Grounding/report severity levels disagree")
    return facts


def build_report_messages(
    *,
    image: str | Path | Any,
    defect_vocabulary: Sequence[str],
    bridge_facts: Mapping[str, Any],
    prompt_config: Mapping[str, Any],
    target_report: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the single shared model-native conversation for train/inference."""

    vocabulary = sorted({str(item) for item in defect_vocabulary})
    if not vocabulary:
        raise ValueError("Applicable dataset defect vocabulary cannot be empty")
    if set(bridge_facts) != set(_BRIDGE_INPUT_KEYS):
        raise ValueError(
            "Bridge textual input must contain exactly the approved factual fields"
        )
    system_instruction = str(prompt_config["system_instruction"]).strip()
    user_instruction = str(prompt_config["user_instruction"]).strip()
    action_note = str(prompt_config["action_supervision_note"]).strip()
    user_text = "\n".join(
        (
            user_instruction,
            f"Applicable defect vocabulary: {compact_json(vocabulary)}",
            f"Authoritative Bridge facts: {compact_json(dict(bridge_facts))}",
            action_note,
        )
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_instruction}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_text},
            ],
        },
    ]
    if target_report is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": serialise_report(target_report)}
                ],
            }
        )
    return messages


def render_prefix_and_full(
    processor: Any,
    *,
    messages_without_target: Sequence[Mapping[str, Any]],
    messages_with_target: Sequence[Mapping[str, Any]],
    target_json: str,
) -> tuple[str, str, str]:
    """Render and prove the native-template assistant boundary."""

    prefix = processor.apply_chat_template(
        list(messages_without_target),
        tokenize=False,
        add_generation_prompt=True,
    )
    full = processor.apply_chat_template(
        list(messages_with_target),
        tokenize=False,
        add_generation_prompt=False,
    )
    if not full.startswith(prefix):
        raise AssertionError("Rendered training prefix is not an exact full-text prefix")
    assistant_suffix = full[len(prefix) :]
    expected_suffix = f"{target_json}<|im_end|>\n"
    if assistant_suffix != expected_suffix:
        raise AssertionError(
            "Native chat-template assistant suffix differs from target plus terminator"
        )
    return prefix, full, assistant_suffix


def prompt_contract_hash(prompt_config: Mapping[str, Any]) -> str:
    """Hash the prompt and serialization contract for checkpoint provenance."""

    payload = {
        "prompt_config": dict(prompt_config),
        "serialization": {
            "sort_keys": True,
            "ensure_ascii": False,
            "allow_nan": False,
            "separators": [",", ":"],
        },
        "bridge_input_keys": list(_BRIDGE_INPUT_KEYS),
    }
    return hashlib.sha256(compact_json(payload).encode("utf-8")).hexdigest()


__all__ = [
    "bridge_prompt_facts",
    "build_report_messages",
    "compact_json",
    "prompt_contract_hash",
    "render_prefix_and_full",
    "serialise_report",
]
