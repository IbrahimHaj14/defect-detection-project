"""Safe JSON persistence and JSON-Schema validation for C3 reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import yaml


def load_json(path: str | Path) -> Any:
    """Load and return one UTF-8 JSON document."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as input_file:
        return json.load(input_file)


def save_json(path: str | Path, obj: Any) -> Path:
    """Atomically write a deterministic UTF-8 JSON document."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
        json.dump(
            obj,
            output_file,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        output_file.write("\n")
    temporary_path.replace(output_path)
    return output_path


def _load_schema(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as schema_file:
        if path.suffix.lower() in {".yaml", ".yml"}:
            schema = yaml.safe_load(schema_file)
        else:
            schema = json.load(schema_file)
    if not isinstance(schema, dict):
        raise ValueError(f"JSON schema must be a mapping: {path}")
    return schema


def validate_against_schema(obj: Any, schema_path: str | Path) -> None:
    """Raise ``jsonschema.ValidationError`` if ``obj`` violates the schema."""

    schema = _load_schema(Path(schema_path))
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(obj)
