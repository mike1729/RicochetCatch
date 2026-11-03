"""Both schema structure and authoritative parser semantics are exercised."""

import io
import json
from pathlib import Path
import zipfile

from jsonschema import Draft202012Validator
import numpy as np
import pytest

import ricochet_catch.assignments as assignments
from ricochet_catch.assignments import (
    AssignmentError,
    load_assignments,
    load_tensors,
    parse_assignments,
    save_tensors,
)
from ricochet_catch.policy import NeuralPolicy, TENSOR_SHAPES, zero_tensors


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "tensor-assignments-v2.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text())
VALIDATOR = Draft202012Validator(SCHEMA)


def document(entries=None):
    return {"format": "tensor-assignments-v2", "explanation": "Fixture only.", "assignments": entries or []}


def entry(tensor="b3", selectors=None, values=1):
    return {"tensor": tensor, "selectors": [[0]] if selectors is None else selectors, "values": values}


def parse_document(value):
    return parse_assignments(json.dumps(value, allow_nan=False))


def test_schema_is_valid_and_empty_submission_is_zero_network() -> None:
    Draft202012Validator.check_schema(SCHEMA)
    VALIDATOR.validate(document())
    parsed = parse_document(document())
    assert parsed.expanded_scalar_writes == parsed.overwritten_entries == parsed.unique_entries == 0
    assert NeuralPolicy(parsed.tensors).act(np.zeros(449, dtype=np.float32)) == 0
    assert parsed.report()["parameter_count"] == 15_555


@pytest.mark.parametrize("envelope", [
    "\n\t {} \r\n",
    "\n```json\n{}\n```\n\t",
    "  ```json\r\n{}\r\n```  ",
])
def test_whitespace_and_exact_optional_envelope(envelope: str) -> None:
    raw = envelope.format(json.dumps(document(), indent=2, ensure_ascii=False))
    assert parse_assignments(raw.encode()).explanation == "Fixture only."


def test_fence_preserves_unicode_separators_inside_json_strings() -> None:
    value = document()
    value["explanation"] = "Unicode paragraph\u2029 and line\u2028 separators are JSON text."
    raw = "```json\r\n" + json.dumps(value, ensure_ascii=False) + "\r\n```"
    assert parse_assignments(raw).explanation == value["explanation"]
    with pytest.raises(AssignmentError):
        parse_assignments("```json\v" + json.dumps(document()) + "\v```")


def test_unsorted_selectors_cartesian_order_and_ordered_overwrites() -> None:
    value = document([
        entry("W1", [[2, 0], [7, 0]], [1, 2, 3, 4]),
        entry("W1", [[0, 2], [0]], [9, 8]),
        entry("b1", [{"start": 1, "stop": 8, "step": 3}], -2.5),
        entry("W1", [[2], [0]], -7),
    ])
    VALIDATOR.validate(value)
    parsed = parse_document(value)
    assert parsed.tensors["W1"][2, 7] == 1
    assert parsed.tensors["W1"][2, 0] == -7
    assert parsed.tensors["W1"][0, 7] == 3
    assert parsed.tensors["W1"][0, 0] == 9
    np.testing.assert_array_equal(parsed.tensors["b1"][[1, 4, 7]], [-2.5] * 3)
    assert np.count_nonzero(parsed.tensors["W1"]) == 4
    assert parsed.expanded_scalar_writes == 10
    assert parsed.overwritten_entries == parsed.overwrite_count == 3
    assert parsed.unique_entries == 7


def test_full_architecture_boundary_and_rounding() -> None:
    entries = []
    for name, shape in TENSOR_SHAPES.items():
        entries.append(entry(name, [{"start": 0, "stop": size, "step": 1} for size in shape], 10_000))
    value = document(entries + [entry("b3", [[2, 0, 1]], [-10_000, 0.1, -0.0])])
    VALIDATOR.validate(value)
    parsed = parse_document(value)
    assert parsed.unique_entries == 15_555
    assert parsed.expanded_scalar_writes == 15_558
    assert parsed.overwritten_entries == 3
    assert all(tensor.dtype == np.float32 for tensor in parsed.tensors.values())
    assert parsed.tensors["b3"][0] == np.float32(0.1)
    assert np.signbit(parsed.tensors["b3"][1])


@pytest.mark.parametrize("raw", [
    '{"format":"tensor-assignments-v2","format":"tensor-assignments-v2","explanation":"","assignments":[]}',
    '{"format":"tensor-assignments-v2","explanation":"","assignments":[{"tensor":"b3","tensor":"b3","selectors":[[0]],"values":0}]}',
    '{"format":"tensor-assignments-v2","explanation":"","assignments":[{"tensor":"b3","selectors":[{"start":0,"start":1,"stop":2,"step":1}],"values":0}]}',
    '{} trailing prose',
    '{} {}',
    '```JSON\n{}\n```',
    '```\n{}\n```',
    '```json\n{}\n```\n```json\n{}\n```',
    'A response:\n```json\n{}\n```',
    '```json\n{}\n```\nExplanation',
    '```json \n{}\n```',
    '```json\n{}\n````',
    '\ufeff{}',
    '[]',
    '',
    b'\xff',
])
def test_invalid_json_keys_or_envelopes_are_rejected(raw) -> None:
    with pytest.raises(AssignmentError):
        parse_assignments(raw)


@pytest.mark.parametrize("bad_entry", [
    entry("unknown"),
    entry(True),
    entry([], values=0),
    entry(selectors=[]),
    entry(selectors=[[0], [1]]),
    entry(selectors=[[True]]),
    entry(selectors=[[0.0]]),
    entry(selectors=[[-1]]),
    entry(selectors=[[3]]),
    entry(selectors=[[]]),
    entry(selectors=[[1, 1]]),
    entry(selectors=[{"start": 0, "stop": 0, "step": 1}]),
    entry(selectors=[{"start": 2, "stop": 1, "step": 1}]),
    entry(selectors=[{"start": 0, "stop": 4, "step": 1}]),
    entry(selectors=[{"start": -1, "stop": 2, "step": 1}]),
    entry(selectors=[{"start": 0, "stop": 2, "step": 0}]),
    entry(selectors=[{"start": 0, "stop": 2, "step": 4}]),
    entry(selectors=[{"start": 0, "stop": 2, "step": True}]),
    entry(selectors=[{"start": 0, "stop": 2, "step": 1.0}]),
    entry(selectors=[{"start": 0, "stop": 2}]),
    entry(selectors=[{"start": 0, "stop": 2, "step": 1, "extra": 0}]),
    entry(values=True),
    entry(values=None),
    entry(values="1"),
    entry(values=[]),
    entry(values=[1, 2]),
    entry(values=[True]),
    entry(values=[[1]]),
    entry(values=10_000.0001),
    entry(values=-10_001),
])
def test_semantically_invalid_assignments_are_rejected(bad_entry) -> None:
    with pytest.raises(AssignmentError):
        parse_document(document([bad_entry]))


@pytest.mark.parametrize("mutate", [
    lambda d: d.update(extra="field"),
    lambda d: d.pop("format"),
    lambda d: d.update(format="tensor-assignments-v1"),
    lambda d: d.update(explanation=[]),
    lambda d: d.update(assignments={}),
    lambda d: d["assignments"].append({**entry(), "extra": 0}),
])
def test_schema_and_parser_both_reject_bad_structure(mutate) -> None:
    value = document()
    mutate(value)
    assert list(VALIDATOR.iter_errors(value))
    with pytest.raises(AssignmentError):
        parse_document(value)


@pytest.mark.parametrize("bad_entry", [
    entry(selectors=[[True]]),
    entry(selectors=[[0, 0]]),
    entry(selectors=[[3]]),
    entry(selectors=[]),
    entry(values=True),
    entry(values=[True]),
    entry(values=10_001),
])
def test_schema_rejects_basic_type_and_bound_errors(bad_entry) -> None:
    assert list(VALIDATOR.iter_errors(document([bad_entry])))


def test_schema_valid_does_not_replace_semantic_validation() -> None:
    wrong_count = document([entry("W1", [[0], [0, 1]], [2])])
    reversed_range = document([entry(selectors=[{"start": 2, "stop": 1, "step": 1}])])
    for value in (wrong_count, reversed_range):
        VALIDATOR.validate(value)
        with pytest.raises(AssignmentError):
            parse_document(value)


@pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity", "1e1000", "-1e1000", "9" * 500])
def test_nonfinite_and_giant_numbers_are_rejected_even_if_later_overwritten(literal: str) -> None:
    raw = json.dumps(document([entry(values="REPLACE"), entry(values=0)])).replace('"REPLACE"', literal)
    with pytest.raises(AssignmentError):
        parse_assignments(raw)


def test_rejecting_late_assignment_is_atomic(monkeypatch) -> None:
    def forbidden_allocation():
        pytest.fail("tensors must not be created before all assignments validate")
    monkeypatch.setattr(assignments, "zero_tensors", forbidden_allocation)
    with pytest.raises(AssignmentError):
        parse_document(document([entry(values=7), entry(values=False)]))


def test_explanation_and_assignment_count_limits() -> None:
    value = document([entry()] * 10_000)
    value["explanation"] = "x" * 10_000
    parsed = parse_document(value)
    assert parsed.overwritten_entries == 9_999
    value["assignments"].append(entry())
    with pytest.raises(AssignmentError):
        parse_document(value)
    value = document()
    value["explanation"] = "x" * 10_001
    with pytest.raises(AssignmentError):
        parse_document(value)
    assert list(VALIDATOR.iter_errors(value))


def test_expanded_write_limit_includes_overwrites() -> None:
    full = entry("W1", [{"start": 0, "stop": 32, "step": 1}, {"start": 0, "stop": 449, "step": 1}])
    # 69 * 14,368 + 32 * 269 = 1,000,000 exactly.
    remainder = entry("W1", [{"start": 0, "stop": 32, "step": 1}, {"start": 0, "stop": 269, "step": 1}])
    value = document([full] * 69 + [remainder])
    parsed = parse_document(value)
    assert parsed.expanded_scalar_writes == 1_000_000
    assert parsed.unique_entries == 14_368
    assert parsed.overwritten_entries == 1_000_000 - 14_368
    value["assignments"].append(entry())
    with pytest.raises(AssignmentError, match="scalar writes"):
        parse_document(value)
    # Absurd range endpoints must be rejected without expansion.
    with pytest.raises(AssignmentError):
        parse_document(document([entry(selectors=[{"start": 0, "stop": 10**100, "step": 1}])]))


def test_response_byte_limit_and_original_file_are_preserved(tmp_path) -> None:
    encoded = json.dumps(document()).encode()
    raw = b" " * (assignments.MAX_RESPONSE_BYTES - len(encoded)) + encoded
    assert parse_assignments(raw).unique_entries == 0
    path = tmp_path / "response.json"
    path.write_bytes(raw)
    assert load_assignments(path).unique_entries == 0
    assert path.read_bytes() == raw
    path.write_bytes(b" " + raw)
    with pytest.raises(AssignmentError, match="5 MiB"):
        load_assignments(path)
    with pytest.raises(AssignmentError):
        parse_assignments("\ud800")


def test_numeric_npz_roundtrip_and_no_pickle(tmp_path) -> None:
    parsed = parse_document(document([entry("W1", [[2, 0], [7, 0]], [1, 2, 3, 4])]))
    path = tmp_path / "weights.npz"
    save_tensors(path, parsed.tensors)
    loaded = load_tensors(path)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(TENSOR_SHAPES)
    for name in TENSOR_SHAPES:
        np.testing.assert_array_equal(loaded[name], parsed.tensors[name])
        assert not loaded[name].flags.writeable


def test_invalid_save_cannot_replace_existing_artifact(tmp_path) -> None:
    path = tmp_path / "weights.npz"
    path.write_bytes(b"original artifact")
    tensors = zero_tensors()
    tensors["b3"] = np.zeros(3, dtype=object)
    with pytest.raises(ValueError):
        save_tensors(path, tensors)
    assert path.read_bytes() == b"original artifact"


@pytest.mark.parametrize("issue", ["missing", "extra", "pickle", "shape", "nonfinite", "forged_header"])
def test_untrusted_tensor_archives_are_rejected(tmp_path, issue: str) -> None:
    tensors = zero_tensors()
    path = tmp_path / "bad.npz"
    if issue == "missing":
        del tensors["W1"]
    elif issue == "extra":
        tensors["extra"] = np.zeros(1, dtype=np.float32)
    elif issue == "pickle":
        tensors["b3"] = np.array([object(), object(), object()], dtype=object)
    elif issue == "shape":
        tensors["b3"] = np.zeros(4, dtype=np.float32)
    elif issue == "nonfinite":
        tensors["b3"][0] = np.nan
    if issue == "forged_header":
        with zipfile.ZipFile(path, "w") as archive:
            for name, shape in TENSOR_SHAPES.items():
                content = io.BytesIO()
                if name == "b3":
                    # Tiny compressed file with an enormous declared shape must
                    # fail its header check before NumPy tries to allocate it.
                    np.lib.format.write_array_header_1_0(content, {"descr": "<f4", "fortran_order": False, "shape": (10**12,)})
                else:
                    np.save(content, tensors[name], allow_pickle=False)
                archive.writestr(f"{name}.npy", content.getvalue())
    else:
        np.savez_compressed(path, **tensors)
    with pytest.raises(AssignmentError):
        load_tensors(path)
