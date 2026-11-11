"""Read-only Comet asset access with a persistent local cache."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from io import BytesIO
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Iterable

import torch


MODEL_RE = re.compile(r"model_(optim|params)_step_(\d+)\.pt$")
ENTRY_RE = re.compile(r"entry_step_(\d+)\.pt$")


@dataclass(frozen=True)
class AssetRef:
    asset_id: str
    file_name: str
    asset_dir: str
    step: int
    size_bytes: int
    kind: str


@dataclass(frozen=True)
class CheckpointPair:
    step: int
    optimizer: AssetRef
    parameters: AssetRef


def create_api(project_root: Path | None = None):
    """Create a Comet API client without embedding credentials in source."""
    try:
        from comet_ml import API
        from dotenv import load_dotenv
    except ImportError as exc:
        raise RuntimeError(
            "Comet dependencies are missing; install the project requirements"
        ) from exc

    if project_root is not None:
        load_dotenv(Path(project_root) / ".env")
    api_key = os.getenv("COMET_API_KEY")
    if not api_key:
        raise RuntimeError("COMET_API_KEY is not set in the environment or .env")
    return API(api_key=api_key)


def get_experiment(api, workspace: str, project: str, key: str):
    experiment = api.get_experiment(
        workspace=workspace,
        project_name=project,
        experiment=key,
    )
    if experiment is None:
        raise ValueError(f"Experiment {workspace}/{project}/{key} was not found")
    return experiment


def experiment_parameters(experiment) -> dict[str, Any]:
    result = {}
    for item in experiment.get_parameters_summary() or []:
        name = item.get("name") or item.get("parameterName")
        if name:
            result[name] = item.get("valueCurrent", item.get("value"))
    return result


def _int_value(value, default=0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _asset_size(asset: dict) -> int:
    for key in ("fileSize", "size", "fileSizeInBytes", "assetSize"):
        if asset.get(key) is not None:
            return _int_value(asset[key])
    return 0


def classify_asset(asset: dict) -> AssetRef | None:
    file_name = str(asset.get("fileName") or "").replace("\\", "/")
    basename = file_name.rsplit("/", 1)[-1]
    model_match = MODEL_RE.search(basename)
    entry_match = ENTRY_RE.search(basename)
    if model_match:
        model_kind, step_text = model_match.groups()
        kind = "model_optim" if model_kind == "optim" else "model_params"
        step = int(step_text)
    elif entry_match:
        kind = "transition"
        step = int(entry_match.group(1))
    else:
        return None
    return AssetRef(
        asset_id=str(asset.get("assetId")),
        file_name=file_name,
        asset_dir=str(asset.get("dir") or ""),
        step=step,
        size_bytes=_asset_size(asset),
        kind=kind,
    )


def list_assets(experiment) -> list[AssetRef]:
    refs = [classify_asset(asset) for asset in experiment.get_asset_list()]
    return sorted(
        (ref for ref in refs if ref is not None),
        key=lambda ref: (ref.kind, ref.step, ref.file_name),
    )


def pair_checkpoints(assets: Iterable[AssetRef]) -> list[CheckpointPair]:
    """Pair optimizer and parameter snapshots only at exactly equal steps."""
    assets = list(assets)
    optim = {asset.step: asset for asset in assets if asset.kind == "model_optim"}
    params = {asset.step: asset for asset in assets if asset.kind == "model_params"}
    return [
        CheckpointPair(step=step, optimizer=optim[step], parameters=params[step])
        for step in sorted(set(optim) & set(params))
    ]


def inventory(experiment, key: str, project: str) -> dict[str, Any]:
    assets = list_assets(experiment)
    pairs = pair_checkpoints(assets)
    by_kind = {}
    for kind in ("model_optim", "model_params", "transition"):
        selected = [asset for asset in assets if asset.kind == kind]
        by_kind[kind] = {
            "count": len(selected),
            "size_bytes": sum(asset.size_bytes for asset in selected),
        }
    return {
        "experiment_key": key,
        "project": project,
        "parameters": experiment_parameters(experiment),
        "asset_summary": by_kind,
        "checkpoint_steps": [pair.step for pair in pairs],
        "unpaired_optimizer_steps": sorted(
            {a.step for a in assets if a.kind == "model_optim"}
            - {p.step for p in pairs}
        ),
        "unpaired_parameter_steps": sorted(
            {a.step for a in assets if a.kind == "model_params"}
            - {p.step for p in pairs}
        ),
    }


def save_inventory(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name.replace("\\", "/"))


def download_asset(experiment, asset: AssetRef, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{asset.step:08d}_{_safe_name(asset.file_name)}"
    if path.exists() and (asset.size_bytes <= 0 or path.stat().st_size == asset.size_bytes):
        return path
    payload = experiment.get_asset(asset.asset_id, return_type="binary")
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return path


def torch_load(path_or_bytes, map_location="cpu"):
    source = BytesIO(path_or_bytes) if isinstance(path_or_bytes, bytes) else path_or_bytes
    try:
        return torch.load(source, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(source, map_location=map_location)


def load_checkpoint_pair(
    experiment,
    pair: CheckpointPair,
    cache_dir: Path,
) -> tuple[dict, dict]:
    optim_path = download_asset(experiment, pair.optimizer, cache_dir / "models")
    params_path = download_asset(experiment, pair.parameters, cache_dir / "models")
    return torch_load(optim_path), torch_load(params_path)


def _cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return value


def _entries(payload) -> list:
    if isinstance(payload, dict) and "memory" in payload:
        return list(payload["memory"])
    if isinstance(payload, list):
        return payload
    return [payload] if isinstance(payload, dict) else []


def _lightweight_entry(entry, source: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, dict):
        fields = getattr(entry, "_fields", ())
        entry = {field: getattr(entry, field) for field in fields}
    keep = (
        "state",
        "next_state",
        "action",
        "action_raw",
        "reward",
        "current_loss",
        "done",
        "reward_model_raw",
        "reward_model",
        "model_reward",
        "opt_model_i",
    )
    result = {key: _cpu_copy(entry[key]) for key in keep if key in entry}
    result["_source"] = source
    return result


def load_lightweight_transitions(
    experiment,
    experiment_key: str,
    project: str,
    cache_dir: Path,
    refresh: bool = False,
    asset_workers: int = 1,
) -> list[dict[str, Any]]:
    """Download transition assets once and cache them without solver models."""
    cache_file = cache_dir / "transitions_light.pt"
    if cache_file.exists() and not refresh:
        return torch_load(cache_file)

    assets = [asset for asset in list_assets(experiment) if asset.kind == "transition"]
    assets = sorted(assets, key=lambda item: item.step)

    asset_cache_dir = cache_dir / "transition_assets"

    def load_one(asset):
        asset_cache = asset_cache_dir / f"{asset.asset_id}.pt"
        if asset_cache.exists() and not refresh:
            return torch_load(asset_cache)

        payload = None
        last_error = None
        for attempt in range(5):
            try:
                payload = experiment.get_asset(
                    asset.asset_id,
                    return_type="binary",
                )
                break
            except Exception as exc:  # Comet wraps requests errors internally.
                last_error = exc
                if attempt == 4:
                    raise
                delay = 2**attempt
                print(
                    f"{experiment_key[:9]} asset step={asset.step} retry "
                    f"{attempt + 1}/4 after {type(exc).__name__}"
                )
                time.sleep(delay)
        if payload is None:
            raise RuntimeError("Asset download failed without a payload") from last_error

        loaded = torch_load(payload)
        asset_entries = []
        for local_index, entry in enumerate(_entries(loaded)):
            asset_entries.append(
                _lightweight_entry(
                    entry,
                    {
                        "project": project,
                        "experiment_key": experiment_key,
                        "asset_step": asset.step,
                        "asset_file": asset.file_name,
                        "local_index": local_index,
                    },
                )
            )
        asset_cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = asset_cache.with_name(f"{asset_cache.name}.{os.getpid()}.part")
        torch.save(asset_entries, temporary)
        temporary.replace(asset_cache)
        return asset_entries

    ordered_entries = [None] * len(assets)
    worker_count = max(1, min(int(asset_workers), len(assets)))
    if worker_count == 1:
        for index, asset in enumerate(assets):
            ordered_entries[index] = load_one(asset)
            completed = index + 1
            if completed % 25 == 0 or completed == len(assets):
                print(
                    f"{experiment_key[:9]} transitions: "
                    f"{completed}/{len(assets)} assets"
                )
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(load_one, asset): index
                for index, asset in enumerate(assets)
            }
            for completed, future in enumerate(as_completed(futures), 1):
                ordered_entries[futures[future]] = future.result()
                if completed % 25 == 0 or completed == len(assets):
                    print(
                        f"{experiment_key[:9]} transitions: "
                        f"{completed}/{len(assets)} assets"
                    )

    transitions = [
        entry
        for asset_entries in ordered_entries
        for entry in asset_entries
    ]
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_file.with_name(
        f"{cache_file.name}.{os.getpid()}.part"
    )
    torch.save(transitions, temporary)
    temporary.replace(cache_file)
    return transitions


def asset_to_dict(asset: AssetRef) -> dict[str, Any]:
    return asdict(asset)


def _metadata(experiment) -> dict[str, Any]:
    try:
        return experiment.get_metadata() or {}
    except Exception:
        return {}


def experiment_key(experiment) -> str:
    metadata = _metadata(experiment)
    return str(
        metadata.get("experimentKey")
        or metadata.get("experiment_key")
        or "unknown"
    )


def select_latest_farm_experiments(
    api,
    workspace: str,
    project: str,
    limit: int,
    minimum_duration_hours: float = 0.07,
):
    """Reproduce the farm loader's duration filter and newest-first order."""
    selected = []
    for experiment in api.get_experiments(workspace=workspace, project_name=project):
        metadata = _metadata(experiment)
        start = _int_value(metadata.get("startTimeMillis"))
        end = _int_value(metadata.get("endTimeMillis"))
        duration = (end - start) / 3_600_000 if start and end else 0.0
        if duration >= minimum_duration_hours:
            selected.append((end, experiment))
    selected.sort(key=lambda item: item[0], reverse=True)
    return [experiment for _, experiment in selected[: int(limit)]]
