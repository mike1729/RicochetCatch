"""Strict, bounded parsing of the declarative tensor-assignments-v2 contract.

No submitted content is executed. Parsing is atomic: no tensor arrays are
allocated or written until the entire document passes semantic validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from types import MappingProxyType
from typing import Any
import zipfile

import numpy as np

from .policy import PARAMETER_COUNT, TENSOR_SHAPES, validate_tensors, zero_tensors


FORMAT = "tensor-assignments-v2"
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_EXPLANATION_CHARS = 10_000
MAX_ASSIGNMENTS = 10_000
MAX_SCALAR_WRITES = 1_000_000
MAX_MAGNITUDE = 10_000


class AssignmentError(ValueError):
    """An entire submission is invalid under the frozen output contract."""


@dataclass(frozen=True)
class AssignmentResult:
    tensors: Mapping[str, np.ndarray]
    explanation: str
    overwritten_entries: int
    expanded_scalar_writes: int
    unique_entries: int

    @property
    def overwrite_count(self) -> int:
        """Number of scalar writes that replaced an earlier assignment."""
        return self.overwritten_entries

    def report(self) -> dict[str, int | str]:
        return {
            "format": FORMAT,
            "explanation": self.explanation,
            "overwritten_entries": self.overwritten_entries,
            "expanded_scalar_writes": self.expanded_scalar_writes,
            "unique_entries": self.unique_entries,
            "parameter_count": PARAMETER_COUNT,
        }


def _reject_constant(value: str) -> None:
    raise AssignmentError(f"nonfinite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AssignmentError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _exact_fields(value: Any, fields: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise AssignmentError(f"{context} must have exactly {', '.join(sorted(fields))}")
    return value


def _integer(value: Any) -> bool:
    # bool is deliberately excluded, including in selectors and ranges.
    return type(value) is int


def _selector(value: Any, axis_size: int, context: str) -> tuple[int, ...]:
    if isinstance(value, list):
        if not 1 <= len(value) <= axis_size:
            raise AssignmentError(f"{context} selection must be nonempty and fit its axis")
        if any(not _integer(i) or not 0 <= i < axis_size for i in value):
            raise AssignmentError(f"{context} has an invalid index")
        if len(set(value)) != len(value):
            raise AssignmentError(f"{context} contains duplicate indices")
        return tuple(value)
    value = _exact_fields(value, {"start", "stop", "step"}, context)
    start, stop, step = value["start"], value["stop"], value["step"]
    if not all(_integer(i) for i in (start, stop, step)):
        raise AssignmentError(f"{context} range values must be integers")
    if not (0 <= start < stop <= axis_size and 1 <= step <= axis_size):
        raise AssignmentError(f"{context} range is out of bounds or empty")
    # Bounds are checked before constructing or expanding a range.
    return tuple(range(start, stop, step))


def _number(value: Any, context: str) -> np.float32:
    if type(value) not in (int, float):
        raise AssignmentError(f"{context} must contain finite JSON numbers, not booleans")
    try:
        if not math.isfinite(value) or abs(value) > MAX_MAGNITUDE:
            raise AssignmentError(f"{context} number is nonfinite or exceeds {MAX_MAGNITUDE}")
        converted = np.float32(value)
    except (OverflowError, TypeError, ValueError) as exc:
        if isinstance(exc, AssignmentError):
            raise
        raise AssignmentError(f"{context} number cannot be converted to float32") from exc
    if not np.isfinite(converted):
        raise AssignmentError(f"{context} number is nonfinite after float32 conversion")
    return converted


def _document(raw: bytes | str) -> Any:
    if isinstance(raw, str):
        try:
            raw_bytes = raw.encode("utf-8")
        except UnicodeError as exc:
            raise AssignmentError("response must be valid UTF-8") from exc
    elif isinstance(raw, bytes):
        raw_bytes = raw
    else:
        raise AssignmentError("response must be bytes or text")
    if len(raw_bytes) > MAX_RESPONSE_BYTES:
        raise AssignmentError("response exceeds the 5 MiB limit")
    try:
        content = raw_bytes.decode("utf-8").strip()
    except UnicodeError as exc:
        raise AssignmentError("response must be valid UTF-8") from exc
    if content.startswith("```"):
        # Inspect only physical LF/CRLF delimiters. str.splitlines() would also
        # split Unicode separators inside valid JSON strings and change content.
        opening, opening_newline, body = content.partition("\n")
        body, closing_newline, closing = body.rpartition("\n")
        if (not opening_newline or not closing_newline
                or opening not in ("```json", "```json\r") or closing != "```"):
            raise AssignmentError("response must use exactly one outer json fence")
        content = body
    try:
        return json.loads(content, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        if isinstance(exc, AssignmentError):
            raise
        raise AssignmentError("response must contain exactly one valid JSON document") from exc


def parse_assignments(raw: bytes | str) -> AssignmentResult:
    """Validate the complete response, then apply assignments in supplied order."""
    document = _exact_fields(_document(raw), {"format", "explanation", "assignments"}, "document")
    if document["format"] != FORMAT:
        raise AssignmentError(f"format must be {FORMAT}")
    explanation = document["explanation"]
    if not isinstance(explanation, str) or len(explanation) > MAX_EXPLANATION_CHARS:
        raise AssignmentError("explanation must be text of at most 10,000 characters")
    assignments = document["assignments"]
    if not isinstance(assignments, list) or len(assignments) > MAX_ASSIGNMENTS:
        raise AssignmentError("assignments must be an array of at most 10,000 entries")

    validated: list[tuple[str, tuple[tuple[int, ...], ...], np.ndarray | np.float32]] = []
    total_writes = 0
    for position, assignment in enumerate(assignments):
        context = f"assignment {position}"
        assignment = _exact_fields(assignment, {"tensor", "selectors", "values"}, context)
        name = assignment["tensor"]
        if not isinstance(name, str) or name not in TENSOR_SHAPES:
            raise AssignmentError(f"{context} has an unknown tensor")
        shape = TENSOR_SHAPES[name]
        selectors = assignment["selectors"]
        if not isinstance(selectors, list) or len(selectors) != len(shape):
            raise AssignmentError(f"{context} has the wrong tensor rank")
        indices = tuple(_selector(selector, size, f"{context} axis {axis}")
                        for axis, (selector, size) in enumerate(zip(selectors, shape)))
        count = math.prod(len(index) for index in indices)
        if total_writes + count > MAX_SCALAR_WRITES:
            raise AssignmentError("expanded scalar writes exceed 1,000,000")
        total_writes += count
        values = assignment["values"]
        if isinstance(values, list):
            if len(values) != count:
                raise AssignmentError(f"{context} has the wrong flat value count")
            converted: np.ndarray | np.float32 = np.asarray(
                [_number(value, context) for value in values], dtype=np.float32
            ).reshape(tuple(len(index) for index in indices))
        else:
            converted = _number(values, context)
        validated.append((name, indices, converted))

    tensors = zero_tensors()
    written = {name: np.zeros(shape, dtype=np.bool_) for name, shape in TENSOR_SHAPES.items()}
    overwritten = 0
    for name, indices, values in validated:
        selection = np.ix_(*indices)
        overwritten += int(np.count_nonzero(written[name][selection]))
        tensors[name][selection] = values
        written[name][selection] = True
    unique_entries = sum(int(np.count_nonzero(mask)) for mask in written.values())
    if unique_entries > PARAMETER_COUNT:  # explicit invariant, independent of write count
        raise AssignmentError("unique entries exceed the fixed parameter count")
    return AssignmentResult(
        MappingProxyType(validate_tensors(tensors)), explanation, overwritten, total_writes, unique_entries
    )


def load_assignments(path: str | os.PathLike[str]) -> AssignmentResult:
    """Read at most limit + 1 bytes. The original response file is never changed."""
    with open(path, "rb") as handle:
        raw = handle.read(MAX_RESPONSE_BYTES + 1)
    return parse_assignments(raw)


def save_tensors(path: str | os.PathLike[str], tensors: Mapping[str, np.ndarray]) -> None:
    """Atomically save only the six validated numeric arrays, without pickle."""
    checked = validate_tensors(tensors)
    destination = Path(path)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.", delete=False) as handle:
            temporary = handle.name
            np.savez_compressed(handle, **checked)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None:
            os.unlink(temporary)


def load_tensors(path: str | os.PathLike[str]) -> dict[str, np.ndarray]:
    """Load a bounded numeric-only NPZ archive and check the fixed architecture."""
    expected = {f"{name}.npy": shape for name, shape in TENSOR_SHAPES.items()}
    try:
        if Path(path).stat().st_size > MAX_RESPONSE_BYTES:
            raise AssignmentError("tensor archive is too large")
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            if len(members) != len(expected) or {member.filename for member in members} != set(expected):
                raise AssignmentError("tensor archive must contain exactly the six tensor arrays")
            for member in members:
                # Inspect decompressed sizes before NumPy can allocate from an
                # untrusted array header. 10 KiB is NumPy's default header cap.
                size_limit = math.prod(expected[member.filename]) * 4 + 10_000
                if member.file_size > size_limit:
                    raise AssignmentError("tensor archive contains an oversized array")
                with archive.open(member) as array_file:
                    version = np.lib.format.read_magic(array_file)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(array_file, max_header_size=10_000)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(array_file, max_header_size=10_000)
                    else:
                        raise AssignmentError("unsupported numeric tensor header")
                    if shape != expected[member.filename] or dtype != np.dtype(np.float32):
                        raise AssignmentError("tensor archive has an invalid shape or dtype")
        with np.load(path, allow_pickle=False, max_header_size=10_000) as archive:
            return validate_tensors({name: archive[name] for name in TENSOR_SHAPES})
    except (OSError, ValueError, KeyError, EOFError, RuntimeError, zipfile.BadZipFile) as exc:
        if isinstance(exc, AssignmentError):
            raise
        raise AssignmentError("invalid numeric tensor archive") from exc
