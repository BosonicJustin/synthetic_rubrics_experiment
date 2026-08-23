"""Target-environment checks that run before a Verl process is launched."""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from compute_as_a_teacher.evaluation.artifacts import (
    canonical_json_bytes,
    file_digest,
    publish_bytes,
    sha256_bytes,
    sha256_text,
)
from compute_as_a_teacher.evaluation.grading import extract_last_boxed
from compute_as_a_teacher.evaluation.prompts import load_prompt

from .anchor_client import OpenAIChatCompletionsClient
from .config import TrainingConfig
from .errors import TrainingError
from .planning import TRAINING_DATA_NAME
from .rewards import compute_math_rewards, render_anchor_prompt
from .verl_adapter import VerlCommand, verify_verl_checkout


PREFLIGHT_NAME = "preflight.json"
_PROBE_MARKER = "CAT_PREFLIGHT_JSON="
_IMAGE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MAX_SAFETENSORS_HEADER_BYTES = 64 * 1024 * 1024
_ANCHOR_CONTEXT_OVERHEAD_BUDGET = 2048
_SAFETENSORS_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "I16": 2,
    "U16": 2,
    "I32": 4,
    "U32": 4,
    "I64": 8,
    "U64": 8,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F16": 2,
    "BF16": 2,
    "F32": 4,
    "F64": 8,
    "C64": 8,
    "C128": 16,
}


def _json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TrainingError(f"Cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingError(f"{name} must be a JSON object: {path}")
    return value


def hash_model_snapshot_tree(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise TrainingError(f"Model snapshot is not a directory: {root}")
    entries: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink() and path.is_dir():
            raise TrainingError(f"Unsupported model snapshot directory symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise TrainingError(f"Unsupported model snapshot entry: {path}")
        digest, size = file_digest(path)
        entries.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": digest,
                "bytes": size,
            }
        )
    if not entries:
        raise TrainingError(f"Model snapshot is empty: {root}")
    return {
        "path": str(root),
        "files": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "tree_sha256": sha256_bytes(canonical_json_bytes(entries)),
        "inventory": entries,
    }


def discover_model_identity(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path).resolve()
    tree = hash_model_snapshot_tree(root)
    tokenizer = _strict_json_object(
        root / "tokenizer_config.json", "tokenizer config"
    )
    template = tokenizer.get("chat_template")
    if not isinstance(template, str) or not template:
        raise TrainingError("Tokenizer config must contain one string chat_template")
    return {
        **tree,
        "snapshot_revision": root.name,
        "chat_template_sha256": sha256_text(template),
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite number {value}")


def _strict_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TrainingError(f"Cannot read {name} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise TrainingError(f"{name} must be a JSON object: {path}")
    return value


def _safetensors_header(path: Path) -> dict[str, Any]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            prefix = handle.read(8)
            if len(prefix) != 8:
                raise ValueError("missing eight-byte header length")
            header_bytes = struct.unpack("<Q", prefix)[0]
            if not 2 <= header_bytes <= _MAX_SAFETENSORS_HEADER_BYTES:
                raise ValueError("header length is outside the supported range")
            if 8 + header_bytes > file_size:
                raise ValueError("header extends beyond the file")
            serialized = handle.read(header_bytes)
        header = json.loads(
            serialized.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise TrainingError(f"Malformed safetensors header {path}: {exc}") from exc
    if not isinstance(header, dict):
        raise TrainingError(f"Malformed safetensors header {path}: expected an object")

    metadata = header.pop("__metadata__", None)
    if metadata is not None and (
        not isinstance(metadata, dict)
        or not all(isinstance(key, str) and isinstance(value, str) for key, value in metadata.items())
    ):
        raise TrainingError(f"Malformed safetensors metadata: {path}")
    if not header:
        raise TrainingError(f"Safetensors file contains no tensors: {path}")

    data_bytes = file_size - 8 - header_bytes
    intervals: list[tuple[int, int, str]] = []
    dtype_counts: dict[str, int] = {}
    for name, descriptor in header.items():
        if not isinstance(name, str) or not name or not isinstance(descriptor, dict):
            raise TrainingError(f"Malformed safetensors tensor descriptor: {path}")
        if set(descriptor) != {"dtype", "shape", "data_offsets"}:
            raise TrainingError(f"Malformed safetensors tensor descriptor {name!r}: {path}")
        dtype = descriptor["dtype"]
        shape = descriptor["shape"]
        offsets = descriptor["data_offsets"]
        if dtype not in _SAFETENSORS_DTYPE_BYTES:
            raise TrainingError(f"Unsupported safetensors dtype {dtype!r}: {path}")
        if not isinstance(shape, list) or any(type(item) is not int or item < 0 for item in shape):
            raise TrainingError(f"Malformed safetensors shape for {name!r}: {path}")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(item) is not int for item in offsets)
        ):
            raise TrainingError(f"Malformed safetensors offsets for {name!r}: {path}")
        start, end = offsets
        if not 0 <= start <= end <= data_bytes:
            raise TrainingError(f"Safetensors offsets are out of bounds for {name!r}: {path}")
        elements = 1
        for dimension in shape:
            elements *= dimension
        if end - start != elements * _SAFETENSORS_DTYPE_BYTES[dtype]:
            raise TrainingError(f"Safetensors byte count disagrees with {name!r}: {path}")
        intervals.append((start, end, name))
        dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

    cursor = 0
    for start, end, name in sorted(intervals):
        if start != cursor:
            raise TrainingError(f"Safetensors data are non-contiguous near {name!r}: {path}")
        cursor = end
    if cursor != data_bytes:
        raise TrainingError(f"Safetensors data do not cover the complete file: {path}")
    return {
        "tensor_names": frozenset(header),
        "tensor_count": len(header),
        "header_bytes": header_bytes,
        "data_bytes": data_bytes,
        "dtypes": dict(sorted(dtype_counts.items())),
    }


def _validate_qwen_safetensors(model_path: Path, weights: Sequence[Path]) -> dict[str, Any]:
    safetensors = sorted(path for path in weights if path.suffix == ".safetensors")
    if not safetensors or len(safetensors) != len(weights):
        raise TrainingError("The pinned Qwen3 snapshot must use only safetensors weights")
    headers = {path.name: _safetensors_header(path) for path in safetensors}
    actual_map: dict[str, str] = {}
    for filename, header in headers.items():
        for tensor_name in header["tensor_names"]:
            if tensor_name in actual_map:
                raise TrainingError(f"Duplicate tensor name across safetensors shards: {tensor_name}")
            actual_map[tensor_name] = filename

    index_path = model_path / "model.safetensors.index.json"
    if len(safetensors) > 1 and not index_path.is_file():
        raise TrainingError("Sharded Qwen3 safetensors require model.safetensors.index.json")
    if index_path.exists():
        index = _strict_json_object(index_path, "safetensors index")
        if set(index) != {"metadata", "weight_map"}:
            raise TrainingError("Safetensors index must contain metadata and weight_map")
        weight_map = index["weight_map"]
        if not isinstance(weight_map, dict) or not weight_map:
            raise TrainingError("Safetensors index weight_map must be nonempty")
        normalized: dict[str, str] = {}
        for tensor_name, filename in weight_map.items():
            if (
                not isinstance(tensor_name, str)
                or not tensor_name
                or not isinstance(filename, str)
                or Path(filename).name != filename
                or not filename.endswith(".safetensors")
            ):
                raise TrainingError("Safetensors index contains an invalid tensor mapping")
            normalized[tensor_name] = filename
        if normalized != actual_map:
            raise TrainingError("Safetensors index does not match shard headers")
        if set(normalized.values()) != set(headers):
            raise TrainingError("Safetensors index does not cover every shard")
        metadata = index["metadata"]
        if not isinstance(metadata, dict):
            raise TrainingError("Safetensors index metadata must be an object")
        total_size = metadata.get("total_size")
        expected_size = sum(header["data_bytes"] for header in headers.values())
        if type(total_size) is not int or total_size != expected_size:
            raise TrainingError("Safetensors index total_size does not match shard data")

    dtypes = sorted(
        {dtype for header in headers.values() for dtype in header["dtypes"]}
    )
    if dtypes != ["BF16"]:
        raise TrainingError(
            f"The pinned BF16 Qwen3 snapshot contains unexpected dtypes: {dtypes}"
        )
    return {
        "index": index_path.name if index_path.is_file() else None,
        "files": len(safetensors),
        "tensors": len(actual_map),
        "data_bytes": sum(header["data_bytes"] for header in headers.values()),
        "dtypes": dtypes,
    }


def _model_files(model_path: Path) -> list[Path]:
    patterns = ("*.safetensors", "pytorch_model*.bin")
    files = sorted({path for pattern in patterns for path in model_path.glob(pattern)})
    valid = []
    for path in files:
        if not path.is_file() or path.stat().st_size == 0:
            raise TrainingError(f"Model weight file is missing or empty: {path}")
        valid.append(path)
    if not valid:
        raise TrainingError("Local model snapshot has no safetensors or PyTorch weights")
    if sum(path.stat().st_size for path in valid) < 1_000_000:
        raise TrainingError("Local model weights are implausibly small")
    index_paths = list(model_path.glob("*.index.json"))
    for index_path in index_paths:
        index = _json_object(index_path, "model weight index")
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise TrainingError(f"Invalid model weight index: {index_path}")
        referenced = {model_path / str(name) for name in weight_map.values()}
        missing = sorted(str(path) for path in referenced if not path.is_file())
        if missing:
            raise TrainingError(f"Model weight index references missing shards: {missing}")
    return valid


def _model_context_length(model_config: Mapping[str, Any]) -> int | None:
    text_config = model_config.get("text_config")
    source = text_config if isinstance(text_config, dict) else model_config
    return next(
        (
            source[key]
            for key in ("max_position_embeddings", "n_positions", "seq_length")
            if type(source.get(key)) is int
        ),
        None,
    )


def inspect_model_snapshot(
    config: TrainingConfig,
    *,
    hash_all_files: bool = False,
) -> dict[str, Any]:
    model_path = Path(config.runtime.model_path).resolve()
    if not model_path.is_dir():
        raise TrainingError(f"Configured local model snapshot is missing: {model_path}")
    if model_path.name not in {
        config.policy.revision,
        config.policy.tokenizer_revision,
    }:
        raise TrainingError(
            "model_path must name the immutable model or tokenizer revision"
        )
    if config.policy.revision != config.policy.tokenizer_revision:
        raise TrainingError(
            "A single local snapshot requires equal model and tokenizer revisions"
        )

    model_config = _json_object(model_path / "config.json", "model config")
    tokenizer_config = _json_object(
        model_path / "tokenizer_config.json", "tokenizer config"
    )
    if not any(
        (model_path / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "vocab.json")
    ):
        raise TrainingError("Local snapshot is missing tokenizer vocabulary files")
    weights = _model_files(model_path)
    is_qwen3 = "qwen3" in config.policy.model_id.lower()
    if is_qwen3 and model_config.get("model_type") != "qwen3":
        raise TrainingError("Configured Qwen3 policy does not contain a Qwen3 model config")
    safetensors = _validate_qwen_safetensors(model_path, weights) if is_qwen3 else None
    quantization = model_config.get("quantization_config")
    if quantization not in (None, {}):
        raise TrainingError("Model config declares quantization for a non-quantized run")
    dtype = model_config.get("dtype", model_config.get("torch_dtype"))
    if dtype is not None and str(dtype).lower().removeprefix("torch.") != "bfloat16":
        raise TrainingError(f"Model config dtype must be bfloat16, found {dtype!r}")
    context_length = _model_context_length(model_config)
    required_policy_context = (
        config.runtime.max_prompt_tokens + config.rollouts.sampling.max_new_tokens
    )
    required_anchor_context = (
        config.rollouts.group_size * config.rollouts.sampling.max_new_tokens
        + config.synthesis.sampling.max_new_tokens
        + _ANCHOR_CONTEXT_OVERHEAD_BUDGET
    )
    required_context = max(required_policy_context, required_anchor_context)
    if context_length is None or context_length < required_context:
        raise TrainingError(
            f"Model context must be at least {required_context}, found {context_length}"
        )
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise TrainingError("Tokenizer config must contain one string chat_template")
    template_sha256 = sha256_text(chat_template)
    if template_sha256 != config.policy.chat_template_sha256:
        raise TrainingError("Local tokenizer chat template does not match the config")

    inventory = [
        {"path": path.name, "bytes": path.stat().st_size}
        for path in weights
    ]
    result: dict[str, Any] = {
        "path": str(model_path),
        "model_revision": config.policy.revision,
        "tokenizer_revision": config.policy.tokenizer_revision,
        "chat_template_sha256": template_sha256,
        "context_length": context_length,
        "required_context_length": required_context,
        "required_policy_context_length": required_policy_context,
        "required_anchor_context_length": required_anchor_context,
        "anchor_context_overhead_budget": _ANCHOR_CONTEXT_OVERHEAD_BUDGET,
        "weight_files": inventory,
        "weight_bytes": sum(item["bytes"] for item in inventory),
        "safetensors": safetensors,
        "all_files_tree_sha256": None,
    }
    if hash_all_files:
        tree = hash_model_snapshot_tree(model_path)
        if tree["tree_sha256"] != config.runtime.model_snapshot_tree_sha256:
            raise TrainingError(
                "Model snapshot tree SHA-256 does not match the preregistered config"
            )
        result["all_files_tree_sha256"] = tree["tree_sha256"]
        result["all_files"] = tree["inventory"]
    return result


_RUNTIME_PROBE = r'''
import ast
import hashlib
import importlib.metadata as md
import importlib.util
import inspect
import json
import os
import platform
import sys
from pathlib import Path

names = ["torch", "ray", "vllm", "transformers", "datasets", "hydra-core", "omegaconf", "flash-attn", "verl"]
packages = {}
for name in names:
    try:
        packages[name] = md.version(name)
    except md.PackageNotFoundError:
        packages[name] = None
all_packages = sorted(
    (dist.metadata.get("Name", "").lower(), dist.version)
    for dist in md.distributions()
    if dist.metadata.get("Name")
)
import torch
from compute_as_a_teacher.training.verl_dataset import JsonlRLHFDataset
from compute_as_a_teacher.training.verl_reward import compute_score

spec = importlib.util.find_spec("verl")
fsdp_worker_path = (
    Path(spec.origin).resolve().parent / "workers" / "fsdp_workers.py"
    if spec and spec.origin
    else None
)
adamw_assignments = 0
fsdp_worker_sha256 = None
if fsdp_worker_path and fsdp_worker_path.is_file():
    fsdp_worker_bytes = fsdp_worker_path.read_bytes()
    fsdp_worker_sha256 = hashlib.sha256(fsdp_worker_bytes).hexdigest()
    fsdp_worker_tree = ast.parse(fsdp_worker_bytes, filename=str(fsdp_worker_path))
    for node in ast.walk(fsdp_worker_tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = {
            target.id for target in node.targets if isinstance(target, ast.Name)
        }
        call = node.value
        if (
            "actor_optimizer" in target_names
            and isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "AdamW"
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "optim"
        ):
            adamw_assignments += 1
cuda_available = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if cuda_available else 0
gpus = []
for index in range(gpu_count):
    free_memory, total_memory = torch.cuda.mem_get_info(index)
    gpus.append({
        "index": index,
        "name": torch.cuda.get_device_name(index),
        "capability": list(torch.cuda.get_device_capability(index)),
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "free_memory_bytes": free_memory,
        "total_memory_bytes": total_memory,
    })
reward_parameters = [
    {
        "name": parameter.name,
        "kind": parameter.kind.name,
        "has_default": parameter.default is not inspect.Parameter.empty,
    }
    for parameter in inspect.signature(compute_score).parameters.values()
]
value = {
    "python": platform.python_version(),
    "executable": sys.executable,
    "platform": platform.platform(),
    "packages": packages,
    "package_inventory": all_packages,
    "package_inventory_sha256": __import__("hashlib").sha256(json.dumps(all_packages, separators=(",", ":")).encode()).hexdigest(),
    "package_count": len(all_packages),
    "torch_cuda_version": torch.version.cuda,
    "cudnn_version": torch.backends.cudnn.version(),
    "nccl_version": torch.cuda.nccl.version() if cuda_available else None,
    "cuda_available": cuda_available,
    "gpu_count": gpu_count,
    "gpus": gpus,
    "verl_module": spec.origin if spec else None,
    "actor_optimizer": {
        "kind": "torch.optim.AdamW" if adamw_assignments == 1 else None,
        "assignment_count": adamw_assignments,
        "source": str(fsdp_worker_path) if fsdp_worker_path else None,
        "source_sha256": fsdp_worker_sha256,
    },
    "trainer_image_digest": os.environ.get("CAT_TRAINER_IMAGE_DIGEST"),
    "custom_modules": {
        "dataset": {
            "module": JsonlRLHFDataset.__module__,
            "name": JsonlRLHFDataset.__name__,
            "source": inspect.getsourcefile(JsonlRLHFDataset),
            "torch_dataset_subclass": issubclass(JsonlRLHFDataset, torch.utils.data.Dataset),
        },
        "reward": {
            "module": compute_score.__module__,
            "name": compute_score.__name__,
            "source": inspect.getsourcefile(compute_score),
            "parameters": reward_parameters,
        },
    },
}
print("CAT_PREFLIGHT_JSON=" + json.dumps(value, sort_keys=True))
'''


def _probe_output(completed: subprocess.CompletedProcess[str], name: str) -> dict[str, Any]:
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-2000:]
        raise TrainingError(f"{name} failed with status {completed.returncode}: {detail}")
    lines = [line for line in completed.stdout.splitlines() if line.startswith(_PROBE_MARKER)]
    if len(lines) != 1:
        raise TrainingError(f"{name} did not emit one structured result")
    try:
        value = json.loads(lines[0][len(_PROBE_MARKER) :])
    except json.JSONDecodeError as exc:
        raise TrainingError(f"{name} emitted invalid JSON") from exc
    if not isinstance(value, dict):
        raise TrainingError(f"{name} result must be an object")
    return value


def _execute_runtime_probe(
    python_executable: str,
    source_path: Path,
    environment: Mapping[str, str],
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]],
) -> dict[str, Any]:
    try:
        completed = runner(
            [python_executable, "-c", _RUNTIME_PROBE],
            cwd=str(source_path),
            env=dict(environment),
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrainingError(f"Cannot run target Python runtime probe: {exc}") from exc
    return _probe_output(completed, "Runtime probe")


def discover_runtime_identity(
    python_executable: str | Path,
    verl_source_path: str | Path,
    repository_root: str | Path,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    python = Path(python_executable).resolve()
    source = Path(verl_source_path).resolve()
    repository = Path(repository_root).resolve()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise TrainingError(f"Target Python is not executable: {python}")
    if not source.is_dir():
        raise TrainingError(f"Verl source path is not a directory: {source}")
    source_root = repository / "src"
    if not source_root.is_dir():
        raise TrainingError(f"Repository source directory is missing: {source_root}")
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONPATH": str(source_root),
        }
    )
    return _execute_runtime_probe(
        str(python), source, environment, runner=runner
    )


def probe_runtime(
    config: TrainingConfig,
    command: VerlCommand,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(command.environment)
    result = _execute_runtime_probe(
        config.runtime.python_executable,
        Path(config.runtime.verl_source_path),
        environment,
        runner=runner,
    )
    packages = result.get("packages")
    if not isinstance(packages, dict):
        raise TrainingError("Runtime probe returned an invalid package inventory")
    missing = [name for name, version in packages.items() if version is None]
    if missing:
        raise TrainingError(f"Runtime packages are missing: {missing}")
    if packages.get("verl") != config.runtime.framework_release:
        raise TrainingError("Installed Verl release does not match the config")
    inventory_digest = result.get("package_inventory_sha256")
    if not isinstance(inventory_digest, str) or not _SHA256.fullmatch(inventory_digest):
        raise TrainingError("Runtime package inventory digest is invalid")
    if inventory_digest != config.runtime.package_inventory_sha256:
        raise TrainingError(
            "Runtime package inventory SHA-256 mismatch: "
            f"expected {config.runtime.package_inventory_sha256}, found {inventory_digest}"
        )
    executable = result.get("executable")
    if not isinstance(executable, str) or Path(executable).resolve() != Path(
        config.runtime.python_executable
    ).resolve():
        raise TrainingError("Runtime probe used a different Python executable")
    if result.get("gpu_count") != config.runtime.gpus_per_node:
        raise TrainingError(
            f"Expected {config.runtime.gpus_per_node} local GPUs, found {result.get('gpu_count')}"
        )
    gpus = result.get("gpus")
    if (
        not isinstance(gpus, list)
        or len(gpus) != config.runtime.gpus_per_node
        or not all(isinstance(gpu, dict) for gpu in gpus)
        or any(
            "h100" not in str(gpu.get("name", "")).lower()
            or gpu.get("bf16_supported") is not True
            for gpu in gpus
        )
    ):
        raise TrainingError("The canonical paper run requires BF16-capable H100 GPUs")
    for gpu in gpus:
        free_memory = gpu.get("free_memory_bytes")
        total_memory = gpu.get("total_memory_bytes")
        if (
            type(free_memory) is not int
            or type(total_memory) is not int
            or not 0 <= free_memory <= total_memory
            or total_memory <= 0
        ):
            raise TrainingError("Runtime probe returned invalid GPU memory data")
        if free_memory / total_memory < config.runtime.minimum_gpu_free_memory_fraction:
            raise TrainingError(
                f"GPU {gpu.get('index')} has insufficient free memory for launch"
            )
    digest = result.get("trainer_image_digest")
    if not isinstance(digest, str) or not _IMAGE_DIGEST.fullmatch(digest):
        raise TrainingError(
            "CAT_TRAINER_IMAGE_DIGEST must contain the immutable trainer image digest"
        )
    if digest != config.runtime.trainer_image_digest:
        raise TrainingError(
            "Trainer image digest mismatch: "
            f"expected {config.runtime.trainer_image_digest}, found {digest}"
        )
    verl_module = result.get("verl_module")
    if not isinstance(verl_module, str):
        raise TrainingError("Runtime probe could not resolve the Verl module")
    try:
        Path(verl_module).resolve().relative_to(Path(config.runtime.verl_source_path).resolve())
    except ValueError as exc:
        raise TrainingError("Runtime imports Verl from outside the pinned checkout") from exc
    actor_optimizer = result.get("actor_optimizer")
    if (
        not isinstance(actor_optimizer, dict)
        or actor_optimizer.get("kind") != "torch.optim.AdamW"
        or actor_optimizer.get("assignment_count") != 1
        or not isinstance(actor_optimizer.get("source"), str)
        or not isinstance(actor_optimizer.get("source_sha256"), str)
        or not _SHA256.fullmatch(actor_optimizer["source_sha256"])
    ):
        raise TrainingError("Pinned Verl runtime does not use AdamW for the FSDP actor")
    try:
        Path(actor_optimizer["source"]).resolve().relative_to(
            Path(config.runtime.verl_source_path).resolve()
        )
    except ValueError as exc:
        raise TrainingError("Actor optimizer source is outside the pinned Verl checkout") from exc
    source_root = command.environment.get("PYTHONPATH")
    if not isinstance(source_root, str) or not source_root or os.pathsep in source_root:
        raise TrainingError("Planned PYTHONPATH must identify one repository source root")
    custom = result.get("custom_modules")
    if not isinstance(custom, dict) or set(custom) != {"dataset", "reward"}:
        raise TrainingError("Runtime did not import both repository custom modules")
    expected_parameters = [
        *( (name, "POSITIONAL_OR_KEYWORD", False) for name in (
            "data_sources", "solution_strs", "ground_truths", "extra_infos"
        )),
        *( (name, "KEYWORD_ONLY", name in {
            "max_answer_chars", "anchor_failure_policy", "anchor_client"
        }) for name in (
            "repository_root", "prompt_path", "prompt_version", "prompt_prefix",
            "anchor_base_url", "anchor_model", "anchor_api_key_env",
            "anchor_timeout_seconds", "anchor_max_concurrency",
            "anchor_temperature", "anchor_top_p", "anchor_top_k",
            "anchor_max_tokens", "base_seed", "max_answer_chars",
            "anchor_failure_policy", "anchor_client",
        )),
    ]
    expected_modules = {
        "dataset": (
            "compute_as_a_teacher.training.verl_dataset",
            "JsonlRLHFDataset",
            Path(source_root) / "compute_as_a_teacher/training/verl_dataset.py",
        ),
        "reward": (
            "compute_as_a_teacher.training.verl_reward",
            "compute_score",
            Path(source_root) / "compute_as_a_teacher/training/verl_reward.py",
        ),
    }
    for role, (module_name, object_name, expected_path) in expected_modules.items():
        value = custom.get(role)
        if (
            not isinstance(value, dict)
            or value.get("module") != module_name
            or value.get("name") != object_name
            or not isinstance(value.get("source"), str)
            or Path(value["source"]).resolve() != expected_path.resolve()
        ):
            raise TrainingError(f"Runtime imported the wrong custom {role} module")
    if custom["dataset"].get("torch_dataset_subclass") is not True:
        raise TrainingError("Custom Verl dataset is not a torch Dataset subclass")
    parameters = custom["reward"].get("parameters")
    normalized_parameters = (
        [
            (item.get("name"), item.get("kind"), item.get("has_default"))
            for item in parameters
        ]
        if isinstance(parameters, list) and all(isinstance(item, dict) for item in parameters)
        else None
    )
    if normalized_parameters != expected_parameters:
        raise TrainingError("Custom Verl reward signature does not match the adapter")
    return result


def compose_verl_config(
    command: VerlCommand,
    *,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update(command.environment)
    try:
        completed = runner(
            [*command.argv, "--cfg", "job", "--resolve"],
            cwd=command.cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrainingError(f"Cannot compose the target Verl config: {exc}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()[-4000:]
        raise TrainingError(
            f"Verl Hydra composition failed with status {completed.returncode}: {detail}"
        )
    payload = completed.stdout.encode("utf-8")
    if not payload.strip():
        raise TrainingError("Verl Hydra composition returned no resolved config")
    return {"bytes": len(payload), "sha256": sha256_bytes(payload)}


_TOKENIZER_PROBE = r'''
import hashlib
import json
import sys
from transformers import AutoTokenizer

(
    model_path,
    data_path,
    max_prompt_tokens,
    anchor_skeleton,
    anchor_canary_template,
    anchor_canary_marker,
) = sys.argv[1:7]
rollout_token_budget, anchor_completion_tokens, boundary_margin = map(int, sys.argv[7:10])
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
rows = [json.loads(line) for line in open(data_path, encoding="utf-8") if line.strip()]
lengths = []
for row in rows:
    lengths.append(len(tokenizer.apply_chat_template(row["prompt"], add_generation_prompt=True)))
template = tokenizer.chat_template
anchor_prompt_overhead = len(tokenizer.apply_chat_template(
    [{"role": "user", "content": anchor_skeleton}],
    add_generation_prompt=True,
))

def anchor_canary(repetitions):
    filler = " x" * repetitions
    message = anchor_canary_template.replace(anchor_canary_marker, filler)
    tokens = len(tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        add_generation_prompt=True,
    ))
    return message, tokens

target_canary_prompt_tokens = (
    anchor_prompt_overhead + rollout_token_budget + boundary_margin
)
low, high = 0, 1
message, canary_prompt_tokens = anchor_canary(high)
while canary_prompt_tokens < target_canary_prompt_tokens:
    low, high = high, high * 2
    if high > rollout_token_budget * 8:
        raise RuntimeError("cannot construct the anchor context canary")
    message, canary_prompt_tokens = anchor_canary(high)
while low + 1 < high:
    middle = (low + high) // 2
    candidate, candidate_tokens = anchor_canary(middle)
    if candidate_tokens < target_canary_prompt_tokens:
        low = middle
    else:
        high = middle
        message, canary_prompt_tokens = candidate, candidate_tokens

value = {
    "rows": len(rows),
    "min_prompt_tokens": min(lengths),
    "max_prompt_tokens": max(lengths),
    "overlong_rows": sum(length > int(max_prompt_tokens) for length in lengths),
    "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
    "tokenizer_class": type(tokenizer).__name__,
    "anchor_prompt_overhead_tokens": anchor_prompt_overhead,
    "anchor_rollout_token_budget": rollout_token_budget,
    "anchor_completion_tokens": anchor_completion_tokens,
    "anchor_boundary_margin_tokens": boundary_margin,
    "required_anchor_context_tokens": (
        anchor_prompt_overhead + rollout_token_budget
        + anchor_completion_tokens + boundary_margin
    ),
    "anchor_context_canary_message": message,
    "anchor_context_canary_prompt_tokens": canary_prompt_tokens,
    "anchor_context_canary_required_tokens": (
        canary_prompt_tokens + anchor_completion_tokens
    ),
}
print("CAT_PREFLIGHT_JSON=" + json.dumps(value, sort_keys=True))
'''


def probe_tokenizer(
    config: TrainingConfig,
    command: VerlCommand,
    run_dir: Path,
    repository_root: Path,
    *,
    expected_rows: int = 500,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if type(expected_rows) is not int or expected_rows <= 0:
        raise TrainingError("Tokenizer probe expected_rows must be a positive integer")
    environment = os.environ.copy()
    environment.update(command.environment)
    template = load_prompt(repository_root, config.synthesis.prompt)
    anchor_skeleton = render_anchor_prompt(
        template,
        config.synthesis.prompt,
        tuple("" for _ in range(config.synthesis.required_rollouts)),
    )
    canary_marker = "CAT_ANCHOR_CONTEXT_FILLER_7F0F4F54"
    canary_answer = sha256_text(
        f"{config.fingerprint}:anchor-context-tail"
    )[:32]
    canary_rollouts = [
        f"Context canary response {index}: {canary_marker}"
        for index in range(config.synthesis.required_rollouts - 1)
    ]
    canary_rollouts.append(
        "Context canary final response: "
        f"{canary_marker}\nThe unique authoritative canary answer is "
        f"\\boxed{{{canary_answer}}}. Preserve it exactly."
    )
    anchor_canary_template = render_anchor_prompt(
        template,
        config.synthesis.prompt,
        tuple(canary_rollouts),
    )
    rollout_token_budget = (
        config.rollouts.group_size * config.rollouts.sampling.max_new_tokens
    )
    anchor_completion_tokens = config.synthesis.sampling.max_new_tokens
    boundary_margin = 2 * config.synthesis.required_rollouts
    try:
        completed = runner(
            [
                config.runtime.python_executable,
                "-c",
                _TOKENIZER_PROBE,
                config.runtime.model_path,
                str(run_dir.resolve() / TRAINING_DATA_NAME),
                str(config.runtime.max_prompt_tokens),
                anchor_skeleton,
                anchor_canary_template,
                canary_marker,
                str(rollout_token_budget),
                str(anchor_completion_tokens),
                str(boundary_margin),
            ],
            cwd=command.cwd,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrainingError(f"Cannot run the tokenizer probe: {exc}") from exc
    result = _probe_output(completed, "Tokenizer probe")
    if result.get("rows") != expected_rows:
        raise TrainingError(
            "Tokenizer probe did not cover the expected "
            f"{expected_rows} training prompts"
        )
    if result.get("overlong_rows") != 0:
        raise TrainingError(
            f"Tokenizer found {result.get('overlong_rows')} overlong prompts"
        )
    if result.get("chat_template_sha256") != config.policy.chat_template_sha256:
        raise TrainingError("Loaded tokenizer chat template does not match the config")
    overhead = result.get("anchor_prompt_overhead_tokens")
    required_anchor_context = result.get("required_anchor_context_tokens")
    context_canary = result.get("anchor_context_canary_message")
    context_canary_tokens = result.get("anchor_context_canary_prompt_tokens")
    context_canary_required = result.get("anchor_context_canary_required_tokens")
    if (
        type(overhead) is not int
        or overhead <= 0
        or result.get("anchor_rollout_token_budget") != rollout_token_budget
        or result.get("anchor_completion_tokens") != anchor_completion_tokens
        or result.get("anchor_boundary_margin_tokens") != boundary_margin
        or required_anchor_context
        != overhead + rollout_token_budget + anchor_completion_tokens + boundary_margin
    ):
        raise TrainingError("Tokenizer returned an invalid synthesis context budget")
    if (
        not isinstance(context_canary, str)
        or not context_canary
        or canary_marker in context_canary
        or context_canary.count(f"\\boxed{{{canary_answer}}}") != 1
        or type(context_canary_tokens) is not int
        or context_canary_tokens
        < overhead + rollout_token_budget + boundary_margin
        or context_canary_required
        != context_canary_tokens + anchor_completion_tokens
    ):
        raise TrainingError("Tokenizer returned an invalid anchor context canary")
    model_context = _model_context_length(
        _json_object(Path(config.runtime.model_path) / "config.json", "model config")
    )
    if model_context is None or required_anchor_context > model_context:
        raise TrainingError(
            "Model context cannot fit eight maximum-length rollouts, the locked "
            f"synthesis prompt, and anchor completion: required {required_anchor_context}, "
            f"found {model_context}"
        )
    if context_canary_required > model_context:
        raise TrainingError(
            "Anchor context canary exceeds the pinned model context: "
            f"required {context_canary_required}, found {model_context}"
        )
    result["required_policy_context_tokens"] = (
        config.runtime.max_prompt_tokens + config.rollouts.sampling.max_new_tokens
    )
    result["anchor_context_canary_expected_answer"] = canary_answer
    result["model_context_tokens"] = model_context
    return result


def probe_anchor(
    config: TrainingConfig,
    repository_root: Path,
    *,
    context_canary_message: str | None = None,
    context_canary_expected_answer: str | None = None,
) -> dict[str, Any]:
    client = OpenAIChatCompletionsClient.from_environment(
        base_url=config.runtime.anchor_base_url,
        api_key_env=config.runtime.anchor_api_key_env,
        timeout_seconds=config.runtime.anchor_timeout_seconds,
    )
    template = load_prompt(repository_root, config.synthesis.prompt)
    rollouts = tuple(
        f"Canary response {index}: final \\boxed{{42}}" for index in range(8)
    )
    prompt = render_anchor_prompt(template, config.synthesis.prompt, rollouts)
    response = client.complete(
        model=config.runtime.anchor_model,
        message=prompt,
        temperature=config.synthesis.sampling.temperature,
        top_p=config.synthesis.sampling.top_p,
        top_k=config.synthesis.sampling.top_k,
        max_tokens=config.synthesis.sampling.max_new_tokens,
        seed=config.synthesis.sampling.base_seed,
    )
    reward = compute_math_rewards(
        rollouts,
        response,
        max_answer_chars=config.reward.max_answer_chars,
        anchor_failure_policy=config.reward.invalid_anchor,
    )
    if (
        reward.anchor_status != "ok"
        or reward.anchor_extraction.value != "42"
        or reward.rewards != (1,) * 8
    ):
        raise TrainingError(
            "Anchor canary did not preserve the unanimous boxed answer"
        )
    result = {
        "endpoint_sha256": sha256_text(client.endpoint),
        "model": config.runtime.anchor_model,
        "prompt_sha256": sha256_text(prompt),
        "response_sha256": sha256_text(response),
        "anchor_extraction_status": reward.anchor_status,
        "unanimous_agreement_rewards": list(reward.rewards),
        "long_context_request_accepted": False,
        "long_context_tail_answer_preserved": False,
    }
    if context_canary_message is not None:
        if (
            not isinstance(context_canary_message, str)
            or not context_canary_message
            or not isinstance(context_canary_expected_answer, str)
            or not context_canary_expected_answer
            or context_canary_message.count(
                f"\\boxed{{{context_canary_expected_answer}}}"
            )
            != 1
        ):
            raise TrainingError("Anchor context canary message is invalid")
        context_response = client.complete(
            model=config.runtime.anchor_model,
            message=context_canary_message,
            temperature=config.synthesis.sampling.temperature,
            top_p=config.synthesis.sampling.top_p,
            top_k=config.synthesis.sampling.top_k,
            max_tokens=config.synthesis.sampling.max_new_tokens,
            seed=config.synthesis.sampling.base_seed,
        )
        context_extraction = extract_last_boxed(
            context_response,
            max_answer_chars=config.reward.max_answer_chars,
        )
        if (
            context_extraction.status != "ok"
            or context_extraction.value != context_canary_expected_answer
        ):
            raise TrainingError(
                "Long-context anchor canary did not preserve the tail boxed answer"
            )
        result.update(
            {
                "long_context_request_accepted": True,
                "long_context_tail_answer_preserved": True,
                "context_prompt_sha256": sha256_text(context_canary_message),
                "context_response_sha256": sha256_text(context_response),
                "context_expected_answer_sha256": sha256_text(
                    context_canary_expected_answer
                ),
            }
        )
    return result


def validate_anchor_probe_result(
    value: Any,
    *,
    expected_model: str,
    require_long_context: bool,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TrainingError("Anchor probe returned an invalid result")
    if (
        value.get("model") != expected_model
        or value.get("anchor_extraction_status") != "ok"
        or value.get("unanimous_agreement_rewards") != [1] * 8
    ):
        raise TrainingError("Anchor probe did not verify unanimous boxed agreement")
    for name in ("endpoint_sha256", "prompt_sha256", "response_sha256"):
        item = value.get(name)
        if not isinstance(item, str) or not _SHA256.fullmatch(item):
            raise TrainingError(f"Anchor probe {name} is invalid")
    if require_long_context:
        if (
            value.get("long_context_request_accepted") is not True
            or value.get("long_context_tail_answer_preserved") is not True
        ):
            raise TrainingError("Anchor probe did not verify the long-context tail canary")
        for name in (
            "context_prompt_sha256",
            "context_response_sha256",
            "context_expected_answer_sha256",
        ):
            item = value.get(name)
            if not isinstance(item, str) or not _SHA256.fullmatch(item):
                raise TrainingError(f"Anchor probe {name} is invalid")
    return dict(value)


def _qualification_lineage(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if value is None:
        return None
    expected = {
        "profile",
        "qualification_fingerprint",
        "training_data_sha256",
        "command_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TrainingError("Qualification preflight lineage has an invalid schema")
    profile = value.get("profile")
    if (
        not isinstance(profile, Mapping)
        or set(profile)
        != {
            "name",
            "prompt_count",
            "prompt_batch_size",
            "max_steps",
            "save_every_steps",
        }
        or not isinstance(profile.get("name"), str)
        or not profile["name"]
        or any(
            type(profile.get(name)) is not int or profile[name] <= 0
            for name in (
                "prompt_count",
                "prompt_batch_size",
                "max_steps",
                "save_every_steps",
            )
        )
    ):
        raise TrainingError("Qualification preflight profile is invalid")
    for name in (
        "qualification_fingerprint",
        "training_data_sha256",
        "command_sha256",
    ):
        item = value.get(name)
        if not isinstance(item, str) or not _SHA256.fullmatch(item):
            raise TrainingError(f"Qualification preflight {name} is invalid")
    return {
        "profile": dict(profile),
        "qualification_fingerprint": value["qualification_fingerprint"],
        "training_data_sha256": value["training_data_sha256"],
        "command_sha256": value["command_sha256"],
    }


def run_preflight(
    config: TrainingConfig,
    manifest: Mapping[str, Any],
    command: VerlCommand,
    run_dir: Path,
    repository_root: Path,
    *,
    hash_model: bool,
    check_anchor: bool,
    qualification_lineage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config.assert_runnable()
    lineage = _qualification_lineage(qualification_lineage)
    expected_rows = lineage["profile"]["prompt_count"] if lineage else 500
    verify_verl_checkout(
        Path(config.runtime.verl_source_path), config.runtime.framework_revision
    )
    model_snapshot = inspect_model_snapshot(config, hash_all_files=hash_model)
    runtime = probe_runtime(config, command)
    hydra_composition = compose_verl_config(command)
    tokenizer_check = dict(
        probe_tokenizer(
            config,
            command,
            run_dir,
            repository_root,
            expected_rows=expected_rows,
        )
    )
    context_canary = tokenizer_check.pop("anchor_context_canary_message", None)
    context_canary_answer = tokenizer_check.pop(
        "anchor_context_canary_expected_answer", None
    )
    if check_anchor and (
        not isinstance(context_canary, str)
        or not isinstance(context_canary_answer, str)
    ):
        raise TrainingError("Tokenizer probe did not produce an anchor context canary")
    anchor_check = (
        validate_anchor_probe_result(
            probe_anchor(
                config,
                repository_root,
                context_canary_message=context_canary,
                context_canary_expected_answer=context_canary_answer,
            ),
            expected_model=config.runtime.anchor_model,
            require_long_context=True,
        )
        if check_anchor
        else None
    )
    checks: dict[str, Any] = {
        "model_snapshot": model_snapshot,
        "runtime": runtime,
        "hydra_composition": hydra_composition,
        "tokenizer": tokenizer_check,
        "anchor": anchor_check,
    }
    missing = []
    if not hash_model:
        missing.append("model content hash")
    if not check_anchor:
        missing.append("anchor canary")
    receipt: dict[str, Any] = {
        "schema_version": 2,
        "kind": "cat_training_preflight",
        "training_plan_fingerprint": manifest["plan_fingerprint"],
        "config_fingerprint": config.fingerprint,
        "command_fingerprint": command.fingerprint,
        "qualification_lineage": lineage,
        "checks": checks,
        "operationally_ready_to_launch": not missing,
        "missing_gates": missing,
        "scientifically_attested": False,
        "attestation_limitations": [
            "anchor_endpoint_weights_and_hardware_are_not_content_attested",
            "anchor_seed_and_top_k_semantics_are_not_content_attested",
            "trainer_image_digest_is_operator_supplied",
        ],
    }
    receipt["preflight_fingerprint"] = sha256_bytes(canonical_json_bytes(receipt))
    return receipt


def write_preflight_receipt(
    run_dir: Path,
    receipt: Mapping[str, Any],
    *,
    force: bool = False,
) -> Path:
    fingerprint = receipt.get("preflight_fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise TrainingError("Preflight receipt fingerprint is invalid")
    unsigned = dict(receipt)
    unsigned.pop("preflight_fingerprint", None)
    if fingerprint != sha256_bytes(canonical_json_bytes(unsigned)):
        raise TrainingError("Preflight receipt fingerprint mismatch")
    payload = canonical_json_bytes(dict(receipt))
    root = run_dir.resolve()
    archive = root / "preflights" / f"{fingerprint}.json"
    publish_bytes(archive, payload)
    path = root / PREFLIGHT_NAME
    publish_bytes(path, payload, force=force)
    return path
