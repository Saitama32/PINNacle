from comet_ml import API
import torch
import io
from dotenv import load_dotenv
import os
import re
from typing import Optional
# === Настройки ===
WORKSPACE = "saitama32"
PROJECT_NAME = "rlpinn"

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
api_key = os.getenv("COMET_API_KEY")


api = API(api_key=api_key)  # или просто API()
# experiment_key = "9da803bf471942d68069d835e2f95651"
DEFAULT_MODEL_ASSET_DIR = "others/rl_model_snapshots"
FALLBACK_OPTIM_ASSET_DIR = "models/rl_agent_optim"
FALLBACK_PARAMS_ASSET_DIR = "models/rl_agent_params"


def _asset_step(asset) -> int:
    match = re.search(r"_step_(\d+)", asset.get("fileName", ""))
    if match:
        return int(match.group(1))

    step = asset.get("step")
    if step is None:
        step = asset.get("metadata", {}).get("step")
    return int(step or 0)


def _normalize_asset_dir(asset_dir: Optional[str]) -> str:
    return (asset_dir or "").replace("\\", "/").strip("/")


def _asset_dir_variants(target_dir: str) -> set[str]:
    normalized = _normalize_asset_dir(target_dir)
    variants = {normalized}
    if normalized.startswith("others/"):
        variants.add(normalized.removeprefix("others/"))
    return variants


def _asset_path(asset) -> str:
    asset_dir = _normalize_asset_dir(asset.get("dir"))
    file_name = _normalize_asset_dir(asset.get("fileName"))
    if asset_dir and file_name:
        return f"{asset_dir}/{file_name}"
    return asset_dir or file_name


def _asset_in_dir(asset, target_dir: str) -> bool:
    asset_dir = _normalize_asset_dir(asset.get("dir"))
    asset_path = _asset_path(asset)
    for candidate_dir in _asset_dir_variants(target_dir):
        if asset_dir == candidate_dir:
            return True
        if asset_path.startswith(f"{candidate_dir}/"):
            return True
    return False


def _format_asset_location(asset) -> str:
    return f"dir='{asset.get('dir')}', fileName='{asset.get('fileName')}'"


def _find_model_assets(assets, optim_dir: str, params_dir: str):
    optim_assets = [
        a for a in assets
        if _asset_in_dir(a, optim_dir)
        and "model_optim_step_" in a.get("fileName", "")
    ]

    params_assets = [
        a for a in assets
        if _asset_in_dir(a, params_dir)
        and "model_params_step_" in a.get("fileName", "")
    ]

    return optim_assets, params_assets


def load_rl_agent_from_comet(
    experiment_key,
    step: Optional[int] = None,
    map_location: str = "cpu",
    asset_dir: str = DEFAULT_MODEL_ASSET_DIR,
):
    """
    Загружает веса RL-агента (model_optim и model_params) из эксперимента Comet ML.
    
    Args:
        rl_agent: экземпляр твоего DQNAgent
        experiment_key (str): ключ эксперимента Comet (например, 'c4e3a8ff9112457d8c674fb68e3817c0')
        step (int|None): шаг, для которого загрузить модели (если None — берётся последний)
        workspace, project: имя workspace и проекта
        api_key: API ключ (если None — берётся из конфига)
    """
    exp = api.get_experiment(workspace=WORKSPACE, project_name=PROJECT_NAME, experiment=experiment_key)
    assets = exp.get_asset_list()

    # --- фильтруем по подпапкам ---
    # --- старая структура ---
    # optim_assets = [a for a in assets if a.get("dir") == "models/rl_agent_optim"]
    # params_assets = [a for a in assets if a.get("dir") == "models/rl_agent_params"]


    # новая схема
    # if not optim_assets or not params_assets:
    optim_assets, params_assets = _find_model_assets(assets, asset_dir, asset_dir)
    selected_asset_dirs = asset_dir

    if not optim_assets or not params_assets:
        optim_assets, params_assets = _find_model_assets(
            assets,
            FALLBACK_OPTIM_ASSET_DIR,
            FALLBACK_PARAMS_ASSET_DIR,
        )
        selected_asset_dirs = f"{FALLBACK_OPTIM_ASSET_DIR}, {FALLBACK_PARAMS_ASSET_DIR}"

    if not optim_assets or not params_assets:
        model_assets = [
            _format_asset_location(a)
            for a in assets
            if "model_optim_step_" in a.get("fileName", "")
            or "model_params_step_" in a.get("fileName", "")
        ][:10]
        sample = "; ".join(model_assets) if model_assets else "no model snapshot assets found"
        raise ValueError(
            "RL model snapshots were not found in Comet asset dirs "
            f"'{asset_dir}' or "
            f"'{FALLBACK_OPTIM_ASSET_DIR}'/'{FALLBACK_PARAMS_ASSET_DIR}'. "
            "Expected files named model_optim_step_<step>.pt and "
            "model_params_step_<step>.pt. "
            f"Sample matching assets: {sample}"
        )

    # --- сортируем по step ---
    print(f"Using Comet model assets from: {selected_asset_dirs}")
    optim_assets.sort(key=_asset_step)
    params_assets.sort(key=_asset_step)

    # --- выбираем нужный шаг ---
    if step is None:
        optim_asset = optim_assets[-1]
        params_asset = params_assets[-1]
        print(f"⬇️ Загружаем последние версии моделей {experiment_key}: step={_asset_step(optim_asset)}/{_asset_step(params_asset)}")
    else:
        # ищем ближайшие по step
        optim_asset = min(optim_assets, key=lambda a: abs(_asset_step(a) - step))
        params_asset = min(params_assets, key=lambda a: abs(_asset_step(a) - step))
        print(f"⬇️ Загружаем модели для шага, ближайшего к {step}: "
              f"optim_step={_asset_step(optim_asset)}, params_step={_asset_step(params_asset)}")

    # === загрузка model_optim ===
    print(f"📦 Загрузка {optim_asset['fileName']} ...")
    optim_bytes = exp.get_asset(optim_asset["assetId"], return_type="binary")
    model_optim_state = torch.load(io.BytesIO(optim_bytes), map_location=map_location)
    print(f"✅ model_optim ({optim_asset['fileName']}) загружен.")

    # === загрузка model_params ===
    print(f"📦 Загрузка {params_asset['fileName']} ...")
    params_bytes = exp.get_asset(params_asset["assetId"], return_type="binary")
    model_params_state = torch.load(io.BytesIO(params_bytes), map_location=map_location)
    print(f"✅ model_params ({params_asset['fileName']}) загружен.")

    print("🎯 Оба state_dict успешно загружены из Comet.")
    return model_optim_state, model_params_state

# if __name__ == "__main__":

#     from tedeous.rl_algorithms import DQNAgent

#     rl_agent = DQNAgent(optimizer_dict=my_opt_dict, device="cuda:0")


    
